use crate::metrics::EngineMetrics;
use crate::model::loader::LoadedModel;
use crate::scheduler::continuous_batching::{Request, Scheduler};
use candle_core::{DType, Device, Result, Tensor};
use rand::Rng;
use std::sync::Arc;
use tokio::sync::{Mutex, Notify};

pub struct Worker {
    pub model: LoadedModel,
    pub scheduler: Arc<Mutex<Scheduler>>,
    pub device: Device,
    pub metrics: Arc<EngineMetrics>,
}

impl Worker {
    pub fn new(
        model: LoadedModel,
        scheduler: Arc<Mutex<Scheduler>>,
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

    pub async fn run_loop(&mut self, notify: Arc<Notify>) -> anyhow::Result<()> {
        loop {
            // Phase 1: Schedule (short lock hold)
            let (to_prefill, to_decode, mut work_batch) = {
                let mut scheduler = self.scheduler.lock().await;
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
                        let chunk_end = (chunk_start
                            + crate::scheduler::continuous_batching::PREFILL_CHUNK_SIZE)
                            .min(req.prompt_tokens.len());

                        if chunk_start < req.prompt_tokens.len() {
                            let chunk_tokens: Vec<f32> = req.prompt_tokens[chunk_start..chunk_end]
                                .iter()
                                .map(|&x| x as f32)
                                .collect();
                            batch.push(WorkItem {
                                req_id: *req_id,
                                input: chunk_tokens,
                                is_last_chunk: chunk_end == req.prompt_tokens.len(),
                                temperature: req.temperature,
                                top_p: req.top_p,
                                is_prefill: true,
                            });
                        } else if req.cached_prefix_len == req.prompt_tokens.len() {
                            let last_token: f32 =
                                req.prompt_tokens.last().map(|&x| x as f32).unwrap_or(0.0);
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
                        let last_token: f32 = req
                            .generated_tokens
                            .last()
                            .map(|&x| x as f32)
                            .unwrap_or(0.0);
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
                let input = Tensor::new(item.input.as_slice(), &self.device)?
                    .unsqueeze(0)?
                    .to_dtype(candle_core::DType::F32)?;

                let logits = match &mut self.model {
                    LoadedModel::Standard(m) => m.forward(&input, 0)?,
                    LoadedModel::Quantized(q) => q.forward(&input, 0)?,
                };

                let next_token = if !logits.dims().is_empty() && logits.dims()[0] > 0 {
                    self.sample(&logits, item.temperature, item.top_p)?
                } else {
                    rand::thread_rng().gen_range(0..100)
                };

                results.push(ComputeResult {
                    req_id: item.req_id,
                    token: next_token,
                    is_last_chunk: item.is_last_chunk,
                });
            }

            // Phase 3: Update scheduler state (short lock hold)
            {
                let mut scheduler = self.scheduler.lock().await;

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
        let dims = logits.dims();

        if dims.is_empty() || dims.iter().all(|&d| d == 0) {
            return Ok(rand::thread_rng().gen_range(0..100));
        }

        let logits = if dims.len() == 1 {
            logits.clone()
        } else {
            logits.get(0)?
        };

        let mut logits = match logits.flatten_all() {
            Ok(l) => l,
            Err(_) => return Ok(rand::thread_rng().gen_range(0..100)),
        };

        if temperature <= 0.0 {
            return if logits.dims()[0] > 0 {
                Ok(logits.argmax(0)?.to_scalar::<u32>()?)
            } else {
                Ok(rand::thread_rng().gen_range(0..100))
            };
        }

        let logits = (&logits / (temperature as f64))?;
        let prs = candle_nn::ops::softmax(&logits, 0)?;
        let mut prs: Vec<f32> = match prs.to_vec1() {
            Ok(p) if !p.is_empty() => p,
            _ => return Ok(rand::thread_rng().gen_range(0..100)),
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
            let mut rng = rand::thread_rng();
            let mut r: f32 = rng.gen::<f32>() * total_p;

            for (id, p) in &indexed_prs {
                r -= p;
                if r <= 0.0 {
                    return Ok(*id as u32);
                }
            }
            Ok(indexed_prs[0].0 as u32)
        } else {
            let mut rng = rand::thread_rng();
            let mut r: f32 = rng.gen::<f32>();
            for (id, &p) in prs.iter().enumerate() {
                r -= p;
                if r <= 0.0 {
                    return Ok(id as u32);
                }
            }
            Ok((prs.len() - 1) as u32)
        }
    }
}

struct WorkItem {
    req_id: u64,
    input: Vec<f32>,
    is_last_chunk: bool,
    temperature: f32,
    top_p: f32,
    is_prefill: bool,
}

struct ComputeResult {
    req_id: u64,
    token: u32,
    is_last_chunk: bool,
}
