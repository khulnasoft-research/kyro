use crate::metrics::EngineMetrics;
use crate::model::kv_cache::{CacheContext, KVCacheManager};
use crate::model::loader::ModelForward;
use crate::scheduler::continuous_batching::Scheduler;
use candle_core::{Device, Result, Tensor};
use rand::Rng;
use std::sync::Arc;
use tokio::sync::{Notify, RwLock};

pub struct Worker {
    pub model: Box<dyn ModelForward + Send>,
    pub scheduler: Arc<RwLock<Scheduler>>,
    pub device: Device,
    pub metrics: Arc<EngineMetrics>,
    pub cache: KVCacheManager,
}

impl Worker {
    pub fn new(
        model: Box<dyn ModelForward + Send>,
        scheduler: Arc<RwLock<Scheduler>>,
        device: Device,
        metrics: Arc<EngineMetrics>,
    ) -> Self {
        let num_kv_heads = 32;
        let head_dim = 128;
        let cache = KVCacheManager::new(num_kv_heads, head_dim);
        Self {
            model,
            scheduler,
            device,
            metrics,
            cache,
        }
    }

    #[tracing::instrument(skip(self, notify))]
    pub async fn run_loop(&mut self, notify: Arc<Notify>) -> anyhow::Result<()> {
        loop {
            let (to_prefill, work_batch) = {
                let mut scheduler = self.scheduler.write().await;
                let (to_prefill, to_decode) = scheduler.schedule();

                if to_prefill.is_empty() && to_decode.is_empty() {
                    drop(scheduler);
                    notify.notified().await;
                    continue;
                }

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
                                top_k: req.top_k,
                                frequency_penalty: req.frequency_penalty,
                                presence_penalty: req.presence_penalty,
                                logit_bias: req.logit_bias.clone(),
                                is_prefill: true,
                                needs_cache_lookup: chunk_start == 0,
                                seed: req.seed,
                            });
                        } else if req.cached_prefix_len == req.prompt_tokens.len() {
                            let last_token = req.prompt_tokens.last().copied().unwrap_or(0);
                            batch.push(WorkItem {
                                req_id: *req_id,
                                input: vec![last_token],
                                is_last_chunk: true,
                                temperature: req.temperature,
                                top_p: req.top_p,
                                top_k: req.top_k,
                                frequency_penalty: req.frequency_penalty,
                                presence_penalty: req.presence_penalty,
                                logit_bias: req.logit_bias.clone(),
                                is_prefill: true,
                                needs_cache_lookup: false,
                                seed: req.seed,
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
                            top_k: req.top_k,
                            frequency_penalty: req.frequency_penalty,
                            presence_penalty: req.presence_penalty,
                            logit_bias: req.logit_bias.clone(),
                            is_prefill: false,
                            needs_cache_lookup: false,
                            seed: req.seed,
                        });
                    }
                }

                (to_prefill, batch)
            };

            for item in &work_batch {
                if item.is_prefill && item.needs_cache_lookup {
                    self.cache.register_request(item.req_id);
                }
            }

            let mut results = Vec::with_capacity(work_batch.len());
            for item in work_batch.iter() {
                let device = &self.device;
                let input = Tensor::new(item.input.as_slice(), device)?.unsqueeze(0)?;

                let mut cache_ctx = CacheContext::new(&mut self.cache, item.req_id);
                let mut logits = self.model.forward(&input, 0, Some(&mut cache_ctx))?;

                // Apply logit bias
                if let Some(ref bias) = item.logit_bias {
                    let vocab_size = logits.dim(logits.dims().len() - 1)?;
                    let mut bias_accum = vec![0.0f32; vocab_size];
                    for (&token_id, &bias_val) in bias {
                        if (token_id as usize) < vocab_size {
                            bias_accum[token_id as usize] += bias_val;
                        }
                    }
                    let bias_t = Tensor::from_slice(&bias_accum, (vocab_size,), device)?;
                    logits = (&logits + &bias_t.unsqueeze(0)?)?;
                }

                // Apply frequency and presence penalties
                if item.frequency_penalty != 0.0 || item.presence_penalty != 0.0 {
                    if let Some(req) = self
                        .scheduler
                        .read()
                        .await
                        .running_queue
                        .iter()
                        .find(|r| r.id == item.req_id)
                    {
                        let mut penalty_mask = vec![0.0f32; logits.dim(logits.dims().len() - 1)?];
                        for &t in &req.generated_tokens {
                            let idx = t as usize;
                            if idx < penalty_mask.len() {
                                penalty_mask[idx] += item.frequency_penalty;
                            }
                        }
                        for &t in &req.generated_tokens {
                            let idx = t as usize;
                            if idx < penalty_mask.len() {
                                penalty_mask[idx] =
                                    penalty_mask[idx].min(0.0) + item.presence_penalty;
                            }
                        }
                        let mask =
                            Tensor::from_slice(&penalty_mask, (penalty_mask.len(),), device)?;
                        logits = (&logits + &mask.unsqueeze(0)?)?;
                    }
                }

                let logits_ref = &logits;
                let next_token = if !logits_ref.dims().is_empty() && logits_ref.dims()[0] > 0 {
                    sample_token(logits_ref, item.temperature, item.top_p, item.top_k)?
                } else {
                    rand::rng().random_range(0..100)
                };

                results.push(ComputeResult {
                    req_id: item.req_id,
                    token: next_token,
                    is_last_chunk: item.is_last_chunk,
                    is_stop: false,
                });
            }

            {
                let mut scheduler = self.scheduler.write().await;

                for res in &mut results {
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

                        // Check stop conditions
                        let gen = &req.generated_tokens;
                        for stop_seq in &req.stop_token_ids {
                            if gen.len() >= stop_seq.len()
                                && gen[gen.len() - stop_seq.len()..] == stop_seq[..]
                            {
                                res.is_stop = true;
                                break;
                            }
                        }
                    }
                }

                for req_id in &to_prefill {
                    scheduler.advance_prefill_cursor(*req_id);
                }

                let finished: Vec<u64> = scheduler
                    .running_queue
                    .iter()
                    .filter(|r| {
                        if r.generated_tokens.len() >= r.max_tokens {
                            return true;
                        }
                        // Check if any stop sequence was matched
                        let gen = &r.generated_tokens;
                        r.stop_token_ids.iter().any(|stop_seq| {
                            gen.len() >= stop_seq.len()
                                && gen[gen.len() - stop_seq.len()..] == stop_seq[..]
                        })
                    })
                    .map(|r| r.id)
                    .collect();

                for id in &finished {
                    scheduler.finish_request(*id);
                    self.cache.unregister_request(*id);
                }
            }

            tokio::task::yield_now().await;
        }
    }
}

pub fn sample_token(
    logits: &Tensor,
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
) -> Result<u32> {
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
    let logits_vec: Vec<f32> = match logits.to_vec1() {
        Ok(v) if !v.is_empty() => v,
        _ => return Ok(rand::rng().random_range(0..100)),
    };

    // Top-K filtering: keep only top K logits
    let mut indexed_logits: Vec<(usize, f32)> = logits_vec.into_iter().enumerate().collect();
    if let Some(k) = top_k {
        if k < indexed_logits.len() {
            indexed_logits.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
            indexed_logits.truncate(k);
        }
    }

    // Apply softmax on filtered logits
    let max_logit = indexed_logits
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let prs: Vec<(usize, f32)> = indexed_logits
        .into_iter()
        .map(|(id, v)| (id, (v - max_logit).exp()))
        .collect();
    let sum_p: f32 = prs.iter().map(|(_, p)| p).sum();
    let prs: Vec<(usize, f32)> = prs.into_iter().map(|(id, p)| (id, p / sum_p)).collect();

    if top_p < 1.0 {
        let mut sorted: Vec<(usize, f32)> = prs.clone();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut cumsum = 0.0;
        let mut cut_off = sorted.len();
        for (i, (_, p)) in sorted.iter().enumerate() {
            cumsum += p;
            if cumsum > top_p {
                cut_off = i + 1;
                break;
            }
        }
        sorted.truncate(cut_off);

        let total_p: f32 = sorted.iter().map(|(_, p)| p).sum();
        let mut rng = rand::rng();
        let mut r: f32 = rng.random::<f32>() * total_p;

        for (id, p) in &sorted {
            r -= p;
            if r <= 0.0 {
                return Ok(*id as u32);
            }
        }
        Ok(sorted[0].0 as u32)
    } else {
        let mut rng = rand::rng();
        let mut r: f32 = rng.random::<f32>();
        for (id, p) in &prs {
            r -= p;
            if r <= 0.0 {
                return Ok(*id as u32);
            }
        }
        Ok(prs.last().map(|(id, _)| *id).unwrap_or(0) as u32)
    }
}

struct WorkItem {
    req_id: u64,
    input: Vec<u32>,
    is_last_chunk: bool,
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
    frequency_penalty: f32,
    presence_penalty: f32,
    logit_bias: Option<std::collections::HashMap<u32, f32>>,
    is_prefill: bool,
    needs_cache_lookup: bool,
    #[allow(dead_code)]
    seed: Option<u64>,
}

struct ComputeResult {
    req_id: u64,
    token: u32,
    #[allow(dead_code)]
    is_last_chunk: bool,
    is_stop: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_sample_argmax_with_zero_temp() {
        let device = Device::Cpu;
        let logits = Tensor::from_slice(
            &[0.1_f32, 0.2, 0.3, 0.4, 0.5, 10.0, 0.1, 0.2],
            &[1, 8],
            &device,
        )
        .unwrap();
        let token = sample_token(&logits, 0.0, 1.0, None).unwrap();
        assert_eq!(token, 5);
    }

    #[test]
    fn test_sample_argmax_2d_no_batch() {
        let device = Device::Cpu;
        let logits = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 0.5, 0.1], &[1, 5], &device).unwrap();
        let token = sample_token(&logits, 0.0, 1.0, None).unwrap();
        assert_eq!(token, 2);
    }

    #[test]
    fn test_sample_1d_logits() {
        let device = Device::Cpu;
        let logits = Tensor::from_slice(&[0.1_f32, 0.2, 5.0, 0.4], &[4], &device).unwrap();
        let token = sample_token(&logits, 0.0, 1.0, None).unwrap();
        assert_eq!(token, 2);
    }

    #[test]
    fn test_sample_with_temperature() {
        let device = Device::Cpu;
        let logits =
            Tensor::from_slice(&[100.0_f32, 0.0, 0.0, 0.0, 0.0], &[1, 5], &device).unwrap();
        let token = sample_token(&logits, 2.0, 1.0, None).unwrap();
        assert_eq!(token, 0);
    }

    #[test]
    fn test_sample_top_p_filters_low_prob_tokens() {
        let device = Device::Cpu;
        let logits =
            Tensor::from_slice(&[100.0_f32, 0.0, 0.0, 0.0, 0.0], &[1, 5], &device).unwrap();
        let token = sample_token(&logits, 1.0, 0.5, None).unwrap();
        assert_eq!(token, 0);
    }

    #[test]
    fn test_sample_top_k_filters() {
        let device = Device::Cpu;
        let logits = Tensor::from_slice(&[1.0_f32, 2.0, 10.0, 0.5, 0.1], &[1, 5], &device).unwrap();
        // top_k=2 should only keep tokens 2 and 1
        let token = sample_token(&logits, 1.0, 1.0, Some(2)).unwrap();
        assert!([1u32, 2].contains(&token));
    }

    #[test]
    fn test_sample_empty_logits_fallback() {
        let device = Device::Cpu;
        let logits = Tensor::from_slice(&[] as &[f32], &[0], &device).unwrap();
        let token = sample_token(&logits, 1.0, 1.0, None).unwrap();
        assert!(token < 100);
    }
}
