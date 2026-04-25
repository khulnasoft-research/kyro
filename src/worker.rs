use std::sync::Arc;
use tokio::sync::Mutex;
use crate::scheduler::continuous_batching::{Scheduler, Request};
use crate::metrics::EngineMetrics;
use crate::model::loader::LoadedModel;
use candle_core::{Device, Tensor, DType, Result};
use rand::distributions::Distribution;

pub struct Worker {
    pub model: LoadedModel,
    pub scheduler: Arc<Mutex<Scheduler>>,
    pub device: Device,
    pub metrics: Arc<EngineMetrics>,
}

impl Worker {
    pub fn new(model: LoadedModel, scheduler: Arc<Mutex<Scheduler>>, device: Device, metrics: Arc<EngineMetrics>) -> Self {
        Self { model, scheduler, device, metrics }
    }

    pub async fn run_loop(&mut self) -> anyhow::Result<()> {
        loop {
            let mut scheduler = self.scheduler.lock().await;
            let (to_prefill, to_decode) = scheduler.schedule();
            
            if to_prefill.is_empty() && to_decode.is_empty() {
                drop(scheduler);
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                continue;
            }

            // 1. Handle Prefills (Chunked + Prefix-Cache-Aware)
            for req_id in to_prefill {
                if let Some(req) = scheduler.running_queue.iter_mut().find(|r| r.id == req_id) {
                    // Only forward the current CHUNK of the prompt (not already-cached prefix)
                    let chunk_start = req.prefill_cursor;
                    let chunk_end = (chunk_start + crate::scheduler::continuous_batching::PREFILL_CHUNK_SIZE)
                        .min(req.prompt_tokens.len());
                    
                    // If chunk_start >= prompt length, the prefix was fully cached — no compute needed
                    if chunk_start < req.prompt_tokens.len() {
                        let chunk_tokens = &req.prompt_tokens[chunk_start..chunk_end];
                        let input = Tensor::new(chunk_tokens, &self.device)?.unsqueeze(0)?;
                        
                        let logits = match &mut self.model {
                            LoadedModel::Standard(m) => m.forward(&input, chunk_start)?,
                            LoadedModel::Quantized(q) => q.forward(&input, chunk_start)?,
                        };

                        // 1.2 Sample next token (only on last chunk)
                        if chunk_end == req.prompt_tokens.len() {
                            let next_token = self.sample(&logits, req.temperature, req.top_p, &mut req.grammar_processor)?;
                            req.generated_tokens.push(next_token);
                            if let Some(sender) = &req.token_sender {
                                let _ = sender.send(next_token);
                            }
                            self.metrics.total_tokens_generated.inc();
                        }
                    } else if req.cached_prefix_len == req.prompt_tokens.len() {
                        // Entire prompt was a cache hit
                        let input = Tensor::new(&[req.prompt_tokens[req.prompt_tokens.len() - 1]], &self.device)?
                            .unsqueeze(0)?.unsqueeze(0)?;
                        let logits = match &mut self.model {
                            LoadedModel::Standard(m) => m.forward(&input, req.prompt_tokens.len() - 1)?,
                            LoadedModel::Quantized(q) => q.forward(&input, req.prompt_tokens.len() - 1)?,
                        };
                        let next_token = self.sample(&logits, req.temperature, req.top_p, &mut req.grammar_processor)?;
                        req.generated_tokens.push(next_token);
                        if let Some(sender) = &req.token_sender {
                            let _ = sender.send(next_token);
                        }
                        self.metrics.total_tokens_generated.inc();
                    }
                }
                // Advance the prefill cursor in the scheduler so the next iteration
                // processes the next chunk or transitions to decode.
                scheduler.advance_prefill_cursor(req_id);
            }

            // 2. Handle Decodes
            for req_id in to_decode {
                if let Some(req) = scheduler.running_queue.iter_mut().find(|r| r.id == req_id) {
                    let timer = self.metrics.token_latency.start_timer();
                    let last_token = *req.generated_tokens.last().unwrap();
                    let input = Tensor::new(&[last_token], &self.device)?.unsqueeze(0)?.unsqueeze(0)?;
                    let index = req.prompt_tokens.len() + req.generated_tokens.len() - 1;
                    
                    let logits = match &mut self.model {
                        LoadedModel::Standard(m) => m.forward(&input, index)?,
                        LoadedModel::Quantized(q) => q.forward(&input, index)?,
                    };
                    let next_token = self.sample(&logits, req.temperature, req.top_p, &mut req.grammar_processor)?;
                    
                    req.generated_tokens.push(next_token);
                    if let Some(sender) = &req.token_sender {
                        let _ = sender.send(next_token);
                    }
                    self.metrics.total_tokens_generated.inc();
                    timer.observe_duration();
                }
            }

            // 3. Cleanup finished requests
            let finished: Vec<u64> = scheduler.running_queue
                .iter()
                .filter(|r| r.generated_tokens.len() >= r.max_tokens)
                .map(|r| r.id)
                .collect();
            
            for id in finished {
                scheduler.finish_request(id);
            }

            drop(scheduler);
            // Yield to other tasks
            tokio::task::yield_now().await;
        }
    }

    /// Performs Temperature + Top-P sampling on the logits.
    fn sample(&self, logits: &Tensor, temperature: f32, top_p: f32, grammar_processor: &mut Option<crate::api::grammar::GrammarLogitsProcessor>) -> Result<u32> {
        let mut logits = logits.narrow(1, logits.dims()[1] - 1, 1)?.squeeze(1)?.squeeze(0)?;
        
        // 1. Apply Grammar Masking (XGrammar)
        if let Some(gp) = grammar_processor {
            let vocab_size = logits.dims()[0];
            logits = gp.apply_grammar_mask(&logits, vocab_size)?;
        }

        if temperature <= 0.0 {
            return Ok(logits.argmax(0)?.to_scalar::<u32>()?);
        }

        // Apply temperature
        let logits = (&logits / (temperature as f64))?;
        let prs = candle_nn::ops::softmax(&logits, 0)?;
        let mut prs: Vec<f32> = prs.to_vec1()?;
        
        // Top-P (Nucleus) Sampling logic
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
            
            // Re-normalize and sample
            let total_p: f32 = indexed_prs.iter().map(|(_, p)| p).sum();
            let mut rng = rand::thread_rng();
            let mut r: f32 = rand::Rng::gen(&mut rng) * total_p;
            
            for (id, p) in indexed_prs {
                r -= p;
                if r <= 0.0 {
                    return Ok(id as u32);
                }
            }
            Ok(indexed_prs[0].0 as u32)
        } else {
            // Standard categorical sampling
            let mut rng = rand::thread_rng();
            let mut r: f32 = rand::Rng::gen(&mut rng);
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
