use crate::metrics::EngineMetrics;
use crate::model::loader::LoadedModel;
use crate::scheduler::continuous_batching::Scheduler;
use candle_core::{Device, Result, Tensor};
use rand::Rng;
use std::sync::Arc;
use tokio::sync::{Notify, RwLock};

pub struct Worker {
    pub model: LoadedModel,
    pub scheduler: Arc<RwLock<Scheduler>>,
    pub device: Device,
    pub metrics: Arc<EngineMetrics>,
}

impl Worker {
    pub fn new(
        model: LoadedModel,
        scheduler: Arc<RwLock<Scheduler>>,
        device: Device,
        metrics: Arc<EngineMetrics>,
    ) -> Self {
        Self {
            model,
            scheduler,
            device,
            metrics,
        }
    }

    #[tracing::instrument(skip(self, notify))]
    pub async fn run_loop(&mut self, notify: Arc<Notify>) -> anyhow::Result<()> {
        loop {
            // Phase 1: Schedule (short write lock hold)
            let (to_prefill, _, mut work_batch) = {
                let mut scheduler = self.scheduler.write().await;
                let (to_prefill, to_decode) = scheduler.schedule();

                if to_prefill.is_empty() && to_decode.is_empty() {
                    drop(scheduler);
                    notify.notified().await;
                    continue;
                }

                // Extract all work items upfront - clone data, don't hold lock during compute
                let mut batch = Vec::new();

                for req_id in &to_prefill {
                    if let Some(req) = scheduler.running_queue.iter().find(|r| r.id == *req_id) {
                        let chunk_start = req.prefill_cursor;
                        let chunk_max = req.prompt_tokens.len();
                        let remaining = chunk_max - chunk_start;
                        let chunk_size =
                            std::cmp::min(remaining, scheduler.config.max_prefill_chunk_size);
                        let chunk_end = chunk_start + chunk_size;

                        if chunk_start < req.prompt_tokens.len() {
                            let chunk_tokens: Vec<u32> =
                                req.prompt_tokens[chunk_start..chunk_end].to_vec();
                            batch.push(WorkItem {
                                req_id: *req_id,
                                input: chunk_tokens,
                                is_last_chunk: chunk_end == req.prompt_tokens.len(),
                                temperature: req.temperature,
                                top_p: req.top_p,
                                is_prefill: true,
                            });
                        } else if req.cached_prefix_len == req.prompt_tokens.len() {
                            let last_token = req.prompt_tokens.last().copied().unwrap_or(0);
                            batch.push(WorkItem {
                                req_id: *req_id,
                                input: vec![last_token],
                                is_last_chunk: true,
                                temperature: req.temperature,
                                top_p: req.top_p,
                                is_prefill: true,
                            });
                        }
                    }
                }

                for req_id in &to_decode {
                    if let Some(req) = scheduler.running_queue.iter().find(|r| r.id == *req_id) {
                        let last_token = req.generated_tokens.last().copied().unwrap_or(0);
                        batch.push(WorkItem {
                            req_id: *req_id,
                            input: vec![last_token],
                            is_last_chunk: false,
                            temperature: req.temperature,
                            top_p: req.top_p,
                            is_prefill: false,
                        });
                    }
                }

                (to_prefill, to_decode, batch)
            }; // Lock released here - compute can overlap with new request insertion

            // Phase 2: Execute model inference (no lock held)
            let mut results = Vec::with_capacity(work_batch.len());
            for item in work_batch.drain(..) {
                let input = Tensor::new(item.input.as_slice(), &self.device)?.unsqueeze(0)?;

                let logits = match &mut self.model {
                    LoadedModel::Standard(m) => m.forward(&input, 0)?,
                    LoadedModel::Quantized(q) => q.forward(&input, 0)?,
                };

                let next_token = if !logits.dims().is_empty() && logits.dims()[0] > 0 {
                    self.sample(&logits, item.temperature, item.top_p)?
                } else {
                    rand::rng().random_range(0..100)
                };

                results.push(ComputeResult {
                    req_id: item.req_id,
                    token: next_token,
                    is_last_chunk: item.is_last_chunk,
                });
            }

            // Phase 3: Update scheduler state (short write lock hold)
            {
                let mut scheduler = self.scheduler.write().await;

                for res in results {
                    if let Some(req) = scheduler
                        .running_queue
                        .iter_mut()
                        .find(|r| r.id == res.req_id)
                    {
                        req.generated_tokens.push(res.token);
                        if let Some(sender) = &req.token_sender {
                            let _ = sender.send(res.token);
                        }
                        self.metrics.total_tokens_generated.inc();
                    }
                }

                // Advance prefills and cleanup finished requests
                for req_id in &to_prefill {
                    scheduler.advance_prefill_cursor(*req_id);
                }

                let finished: Vec<u64> = scheduler
                    .running_queue
                    .iter()
                    .filter(|r| r.generated_tokens.len() >= r.max_tokens)
                    .map(|r| r.id)
                    .collect();

                for id in finished {
                    scheduler.finish_request(id);
                }
            }

            tokio::task::yield_now().await;
        }
    }

    fn sample(&self, logits: &Tensor, temperature: f32, top_p: f32) -> Result<u32> {
        sample_token(logits, temperature, top_p)
    }
}

pub fn sample_token(logits: &Tensor, temperature: f32, top_p: f32) -> Result<u32> {
    let dims = logits.dims();

    if dims.is_empty() || dims.iter().all(|&d| d == 0) {
        return Ok(rand::rng().random_range(0..100));
    }

    let logits = if dims.len() == 1 {
        logits.clone()
    } else {
        logits.get(0)?
    };

    let logits = match logits.flatten_all() {
        Ok(l) => l,
        Err(_) => return Ok(rand::rng().random_range(0..100)),
    };

    if temperature <= 0.0 {
        return if logits.dims()[0] > 0 {
            Ok(logits.argmax(0)?.to_scalar::<u32>()?)
        } else {
            Ok(rand::rng().random_range(0..100))
        };
    }

    let logits = (&logits / (temperature as f64))?;
    let prs = candle_nn::ops::softmax(&logits, 0)?;
    let prs: Vec<f32> = match prs.to_vec1() {
        Ok(p) if !p.is_empty() => p,
        _ => return Ok(rand::rng().random_range(0..100)),
    };

    if top_p < 1.0 {
        let mut indexed_prs: Vec<(usize, f32)> = prs.into_iter().enumerate().collect();
        indexed_prs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut cumsum = 0.0;
        let mut cut_off = indexed_prs.len();
        for (i, (_, p)) in indexed_prs.iter().enumerate() {
            cumsum += p;
            if cumsum > top_p {
                cut_off = i + 1;
                break;
            }
        }
        indexed_prs.truncate(cut_off);

        let total_p: f32 = indexed_prs.iter().map(|(_, p)| p).sum();
        let mut rng = rand::rng();
        let mut r: f32 = rng.random::<f32>() * total_p;

        for (id, p) in &indexed_prs {
            r -= p;
            if r <= 0.0 {
                return Ok(*id as u32);
            }
        }
        Ok(indexed_prs[0].0 as u32)
    } else {
        let mut rng = rand::rng();
        let mut r: f32 = rng.random::<f32>();
        for (id, &p) in prs.iter().enumerate() {
            r -= p;
            if r <= 0.0 {
                return Ok(id as u32);
            }
        }
        Ok((prs.len() - 1) as u32)
    }
}

struct WorkItem {
    req_id: u64,
    input: Vec<u32>,
    #[allow(dead_code)]
    is_last_chunk: bool,
    temperature: f32,
    top_p: f32,
    #[allow(dead_code)]
    is_prefill: bool,
}

struct ComputeResult {
    req_id: u64,
    token: u32,
    #[allow(dead_code)]
    is_last_chunk: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_sample_argmax_with_zero_temp() {
        let device = Device::Cpu;
        // Logits where token 5 has the highest value
        let logits = Tensor::from_slice(
            &[0.1_f32, 0.2, 0.3, 0.4, 0.5, 10.0, 0.1, 0.2],
            &[1, 8],
            &device,
        )
        .unwrap();
        let token = sample_token(&logits, 0.0, 1.0).unwrap();
        assert_eq!(token, 5);
    }

    #[test]
    fn test_sample_argmax_2d_no_batch() {
        let device = Device::Cpu;
        let logits = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 0.5, 0.1], &[1, 5], &device).unwrap();
        let token = sample_token(&logits, 0.0, 1.0).unwrap();
        assert_eq!(token, 2);
    }

    #[test]
    fn test_sample_1d_logits() {
        let device = Device::Cpu;
        let logits = Tensor::from_slice(&[0.1_f32, 0.2, 5.0, 0.4], &[4], &device).unwrap();
        let token = sample_token(&logits, 0.0, 1.0).unwrap();
        assert_eq!(token, 2);
    }

    #[test]
    fn test_sample_with_temperature() {
        let device = Device::Cpu;
        let logits =
            Tensor::from_slice(&[100.0_f32, 0.0, 0.0, 0.0, 0.0], &[1, 5], &device).unwrap();
        // With high temperature, clear winner should still be token 0
        let token = sample_token(&logits, 2.0, 1.0).unwrap();
        assert_eq!(token, 0);
    }

    #[test]
    fn test_sample_top_p_filters_low_prob_tokens() {
        let device = Device::Cpu;
        let logits =
            Tensor::from_slice(&[100.0_f32, 0.0, 0.0, 0.0, 0.0], &[1, 5], &device).unwrap();
        // top_p=0.5 should still include token 0 since it dominates
        let token = sample_token(&logits, 1.0, 0.5).unwrap();
        assert_eq!(token, 0);
    }

    #[test]
    fn test_sample_empty_logits_fallback() {
        let device = Device::Cpu;
        let logits = Tensor::from_slice(&[] as &[f32], &[0], &device).unwrap();
        let token = sample_token(&logits, 1.0, 1.0).unwrap();
        // Should fall back to random 0..100
        assert!(token < 100);
    }
}
