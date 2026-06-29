use std::collections::HashMap;

use candle_core::{Result, Tensor};
use candle_nn::ops::log_softmax;

use crate::model::kv_cache::CacheContext;
use crate::model::loader::{LoadedModel, ModelForward};

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Beam {
    pub tokens: Vec<u32>,
    pub score: f64,
    pub finished: bool,
}

impl Beam {
    #[allow(dead_code)]
    pub fn new(prompt: Vec<u32>) -> Self {
        Self {
            tokens: prompt,
            score: 0.0,
            finished: false,
        }
    }
}

#[allow(dead_code)]
pub struct BeamSearchDecoder {
    pub model: Box<dyn ModelForward + Send>,
    pub num_beams: usize,
    pub length_penalty: f64,
    pub max_new_tokens: usize,
    pub eos_token_id: u32,
}

impl BeamSearchDecoder {
    #[allow(dead_code)]
    pub fn new(
        model: Box<dyn ModelForward + Send>,
        num_beams: usize,
        max_new_tokens: usize,
        eos_token_id: u32,
    ) -> Self {
        Self {
            model,
            num_beams,
            length_penalty: 1.0,
            max_new_tokens,
            eos_token_id,
        }
    }

    #[allow(dead_code)]
    pub fn with_length_penalty(mut self, penalty: f64) -> Self {
        self.length_penalty = penalty;
        self
    }

    #[allow(dead_code)]
    pub fn decode(&mut self, prompt: &[u32]) -> Result<Vec<u32>> {
        let device = &candle_core::Device::Cpu;
        let mut beams: Vec<Beam> = (0..self.num_beams)
            .map(|_| Beam::new(prompt.to_vec()))
            .collect();

        for _step in 0..self.max_new_tokens {
            let mut all_candidates: Vec<(f64, Vec<u32>)> = Vec::new();

            for beam in beams.iter_mut() {
                if beam.finished {
                    continue;
                }

                let input_f32: Vec<f32> = beam.tokens.iter().map(|&t| t as f32).collect();
                let input = Tensor::from_slice(&input_f32, (1, 1, beam.tokens.len()), device)?;
                let logits = self.model.forward(&input, 0, None)?;
                let last_logits = logits.squeeze(1)?.squeeze(0)?;

                let log_probs = log_softmax(&last_logits, 0)?;
                let mut scores: Vec<(f64, u32)> = log_probs
                    .to_vec1::<f32>()?
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (v as f64, i as u32))
                    .collect();
                scores.sort_unstable_by(|a, b| {
                    b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                });

                for &(log_prob, token_id) in scores.iter().take(self.num_beams) {
                    let mut new_tokens = beam.tokens.clone();
                    new_tokens.push(token_id);
                    let len = new_tokens.len() as f64;
                    let score = (beam.score + log_prob) / len.powf(self.length_penalty);
                    all_candidates.push((score, new_tokens));
                }
            }

            if all_candidates.is_empty() {
                break;
            }

            all_candidates.sort_unstable_by(|a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            all_candidates.truncate(self.num_beams);

            beams.clear();
            for (score, tokens) in all_candidates {
                let finished = tokens.last() == Some(&self.eos_token_id);
                beams.push(Beam {
                    tokens,
                    score,
                    finished,
                });
            }

            if beams.iter().all(|b| b.finished) {
                break;
            }
        }

        beams.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(beams
            .into_iter()
            .next()
            .map(|b| b.tokens)
            .unwrap_or_else(|| prompt.to_vec()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum DraftStrategy {
    /// Use a smaller draft model (classic speculative decoding)
    DraftModel,
    /// N-gram matching from the prompt context
    Ngram { order: usize },
    /// Parallel decoding: predict K tokens from the same hidden state (Medusa-style)
    Parallel,
}

#[allow(dead_code)]
pub struct SpeculativeDecoder {
    pub target_model: LoadedModel,
    pub draft_model: Option<LoadedModel>,
    pub lookahead: usize,
    pub strategy: DraftStrategy,
    pub ngram_table: HashMap<Vec<u32>, u32>,
}

impl SpeculativeDecoder {
    #[allow(dead_code)]
    pub fn new(
        target_model: LoadedModel,
        draft_model: Option<LoadedModel>,
        lookahead: usize,
    ) -> Self {
        Self {
            target_model,
            draft_model,
            lookahead,
            strategy: DraftStrategy::DraftModel,
            ngram_table: HashMap::new(),
        }
    }

    #[allow(dead_code)]
    pub fn with_ngram(target_model: LoadedModel, lookahead: usize, order: usize) -> Self {
        Self {
            target_model,
            draft_model: None,
            lookahead,
            strategy: DraftStrategy::Ngram { order },
            ngram_table: HashMap::new(),
        }
    }

    #[allow(dead_code)]
    pub fn with_parallel(target_model: LoadedModel, lookahead: usize) -> Self {
        Self {
            target_model,
            draft_model: None,
            lookahead,
            strategy: DraftStrategy::Parallel,
            ngram_table: HashMap::new(),
        }
    }

    /// Build n-gram table from a corpus of token sequences.
    #[allow(dead_code)]
    pub fn build_ngram_table(&mut self, corpus: &[Vec<u32>], order: usize) {
        self.ngram_table.clear();
        for seq in corpus {
            if seq.len() < order {
                continue;
            }
            for i in 0..=seq.len() - order {
                let prefix = seq[i..i + order - 1].to_vec();
                let next = seq[i + order - 1];
                self.ngram_table.insert(prefix, next);
            }
        }
    }

    /// Generate draft tokens using the selected strategy.
    fn generate_drafts(
        &mut self,
        input: &Tensor,
        index: usize,
        #[allow(unused_mut)] mut cache: Option<&mut CacheContext>,
        context_tokens: &[u32],
    ) -> Result<Vec<u32>> {
        match self.strategy {
            DraftStrategy::DraftModel => {
                let model = self
                    .draft_model
                    .as_mut()
                    .ok_or_else(|| candle_core::Error::Msg("draft model required".into()))?;
                let device = input.device();
                let mut draft_tokens = Vec::with_capacity(self.lookahead);
                let mut current_input = input.clone();

                for i in 0..self.lookahead {
                    #[allow(clippy::needless_option_as_deref)]
                    let logits = model.forward(&current_input, index + i, cache.as_deref_mut())?;
                    let next_token = logits
                        .squeeze(1)?
                        .squeeze(0)?
                        .argmax(0)?
                        .to_scalar::<u32>()?;
                    draft_tokens.push(next_token);
                    current_input = Tensor::new(&[next_token], device)?
                        .unsqueeze(0)?
                        .unsqueeze(0)?;
                }
                Ok(draft_tokens)
            }
            DraftStrategy::Ngram { order } => {
                let mut draft_tokens = Vec::with_capacity(self.lookahead);
                let mut ctx = context_tokens.to_vec();

                for _ in 0..self.lookahead {
                    if ctx.len() < order {
                        break;
                    }
                    let prefix = ctx[ctx.len() - (order - 1)..].to_vec();
                    match self.ngram_table.get(&prefix) {
                        Some(&next) => {
                            draft_tokens.push(next);
                            ctx.push(next);
                        }
                        None => break,
                    }
                }
                Ok(draft_tokens)
            }
            DraftStrategy::Parallel => {
                // Parallel speculation: target model predicts multiple tokens
                // from the same hidden state (like Medusa heads).
                // For now, use a simple repetition: repeat the last token.
                let device = input.device();
                let last_token = context_tokens.last().copied().unwrap_or(0);
                let repeated = vec![last_token; self.lookahead];
                let draft_tensor = Tensor::new(repeated.as_slice(), device)?.unsqueeze(0)?;
                #[allow(clippy::needless_option_as_deref)]
                let logits =
                    self.target_model
                        .forward(&draft_tensor, index, cache.as_deref_mut())?;
                let mut draft_tokens = Vec::with_capacity(self.lookahead);
                for i in 0..self.lookahead {
                    let token = logits.get(i)?.argmax(0)?.to_scalar::<u32>()?;
                    draft_tokens.push(token);
                }
                Ok(draft_tokens)
            }
        }
    }

    /// Rejection sampling to verify draft tokens.
    fn verify_drafts(
        &mut self,
        input: &Tensor,
        index: usize,
        #[allow(unused_mut)] mut cache: Option<&mut CacheContext>,
        draft_tokens: &[u32],
    ) -> Result<Vec<u32>> {
        let device = input.device();
        let num_drafts = draft_tokens.len();

        if num_drafts == 0 {
            #[allow(clippy::needless_option_as_deref)]
            let logits = self.target_model.forward(input, index, cache)?;
            let token = logits
                .squeeze(1)?
                .squeeze(0)?
                .argmax(0)?
                .to_scalar::<u32>()?;
            return Ok(vec![token]);
        }

        let draft_slice: Vec<u32> = draft_tokens.to_vec();
        let draft_tensor = Tensor::new(draft_slice.as_slice(), device)?.unsqueeze(0)?;
        #[allow(clippy::needless_option_as_deref)]
        let target_logits =
            self.target_model
                .forward(&draft_tensor, index, cache.as_deref_mut())?;

        let mut accepted = Vec::new();
        for (i, &draft_token) in draft_tokens.iter().enumerate() {
            let target_top = target_logits.get(i)?.argmax(0)?.to_scalar::<u32>()?;
            if target_top == draft_token {
                accepted.push(draft_token);
            } else {
                accepted.push(target_top);
                break;
            }
        }

        if accepted.is_empty() {
            #[allow(clippy::needless_option_as_deref)]
            let logits = self
                .target_model
                .forward(input, index, cache.as_deref_mut())?;
            let token = logits
                .squeeze(1)?
                .squeeze(0)?
                .argmax(0)?
                .to_scalar::<u32>()?;
            accepted.push(token);
        }

        Ok(accepted)
    }

    #[allow(dead_code)]
    pub fn step(
        &mut self,
        input: &Tensor,
        index: usize,
        #[allow(unused_mut)] mut cache: Option<&mut CacheContext>,
        context_tokens: &[u32],
    ) -> Result<Vec<u32>> {
        let draft_tokens =
            self.generate_drafts(input, index, cache.as_deref_mut(), context_tokens)?;
        #[allow(clippy::needless_option_as_deref)]
        self.verify_drafts(input, index, cache.as_deref_mut(), &draft_tokens)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device};

    use super::*;
    use crate::model::config::LlamaConfig;
    use crate::model::llama::LlamaModel;
    use crate::model::model_registry::ModelInstance;

    fn make_dummy_decoder(lookahead: usize) -> SpeculativeDecoder {
        let cfg = LlamaConfig::llama_7b();
        let model = ModelInstance::Llama(LlamaModel::dummy(&cfg).unwrap());
        let draft = ModelInstance::Llama(LlamaModel::dummy(&cfg).unwrap());
        SpeculativeDecoder {
            target_model: LoadedModel::Standard(model),
            draft_model: Some(LoadedModel::Standard(draft)),
            lookahead,
            strategy: DraftStrategy::DraftModel,
            ngram_table: HashMap::new(),
        }
    }

    #[test]
    fn test_speculative_decoder_creation() {
        let decoder = make_dummy_decoder(5);
        assert_eq!(decoder.lookahead, 5);
    }

    #[test]
    fn test_speculative_step_returns_tokens() {
        let device = Device::Cpu;
        let mut decoder = make_dummy_decoder(1);
        let input = Tensor::zeros((1, 1, 1), DType::F32, &device).unwrap();
        let result = decoder.step(&input, 0, None, &[]).unwrap();
        assert!(!result.is_empty(), "should produce at least one token");
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_ngram_strategy_creation() {
        let cfg = LlamaConfig::llama_7b();
        let model = ModelInstance::Llama(LlamaModel::dummy(&cfg).unwrap());
        let decoder = SpeculativeDecoder::with_ngram(LoadedModel::Standard(model), 3, 4);
        assert_eq!(decoder.lookahead, 3);
        assert_eq!(decoder.strategy, DraftStrategy::Ngram { order: 4 });
        assert!(decoder.draft_model.is_none());
    }

    #[test]
    fn test_ngram_table_building() {
        let cfg = LlamaConfig::llama_7b();
        let model = ModelInstance::Llama(LlamaModel::dummy(&cfg).unwrap());
        let mut decoder = SpeculativeDecoder::with_ngram(LoadedModel::Standard(model), 3, 2);
        let corpus = vec![vec![1, 2, 3, 4, 5]];
        decoder.build_ngram_table(&corpus, 2);
        assert_eq!(decoder.ngram_table.get(&[1][..]), Some(&2));
        assert_eq!(decoder.ngram_table.get(&[2][..]), Some(&3));
        assert_eq!(decoder.ngram_table.get(&[3][..]), Some(&4));
    }

    #[test]
    fn test_parallel_strategy_creation() {
        let cfg = LlamaConfig::llama_7b();
        let model = ModelInstance::Llama(LlamaModel::dummy(&cfg).unwrap());
        let decoder = SpeculativeDecoder::with_parallel(LoadedModel::Standard(model), 4);
        assert_eq!(decoder.strategy, DraftStrategy::Parallel);
        assert_eq!(decoder.lookahead, 4);
    }

    #[test]
    fn test_draft_strategy_equality() {
        assert_eq!(DraftStrategy::DraftModel, DraftStrategy::DraftModel);
        assert_eq!(
            DraftStrategy::Ngram { order: 3 },
            DraftStrategy::Ngram { order: 3 }
        );
        assert_ne!(
            DraftStrategy::Ngram { order: 3 },
            DraftStrategy::Ngram { order: 4 }
        );
    }

    #[test]
    fn test_beam_creation() {
        let beam = Beam::new(vec![1, 2, 3]);
        assert_eq!(beam.tokens, vec![1, 2, 3]);
        assert!(!beam.finished);
        assert_eq!(beam.score, 0.0);
    }

    #[test]
    fn test_beam_search_decoder_creation() {
        let cfg = LlamaConfig::llama_7b();
        let model = ModelInstance::Llama(LlamaModel::dummy(&cfg).unwrap());
        let decoder = BeamSearchDecoder::new(Box::new(model), 4, 50, 2);
        assert_eq!(decoder.num_beams, 4);
        assert_eq!(decoder.max_new_tokens, 50);
        assert_eq!(decoder.eos_token_id, 2);
    }

    #[test]
    fn test_beam_search_with_length_penalty() {
        let cfg = LlamaConfig::llama_7b();
        let model = ModelInstance::Llama(LlamaModel::dummy(&cfg).unwrap());
        let decoder = BeamSearchDecoder::new(Box::new(model), 3, 10, 2).with_length_penalty(0.6);
        assert_eq!(decoder.length_penalty, 0.6);
    }

    #[test]
    fn test_beam_search_decode_returns_tokens() {
        let cfg = LlamaConfig::llama_7b();
        let model = ModelInstance::Llama(LlamaModel::dummy(&cfg).unwrap());
        let mut decoder = BeamSearchDecoder::new(Box::new(model), 2, 5, 2);
        let result = decoder.decode(&[1, 2, 3]).unwrap();
        assert!(result.len() >= 3, "should keep prompt tokens");
    }
}
