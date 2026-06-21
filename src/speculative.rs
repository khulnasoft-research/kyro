use crate::model::loader::LoadedModel;
use candle_core::{Result, Tensor};

#[allow(dead_code)]
pub struct SpeculativeDecoder {
    pub target_model: LoadedModel,
    pub draft_model: LoadedModel,
    pub lookahead: usize,
}

impl SpeculativeDecoder {
    #[allow(dead_code)]
    pub fn new(target_model: LoadedModel, draft_model: LoadedModel, lookahead: usize) -> Self {
        Self {
            target_model,
            draft_model,
            lookahead,
        }
    }

    /// Runs speculative decoding: draft model generates `lookahead` tokens,
    /// target model verifies them in a single forward pass.
    #[allow(dead_code)]
    pub fn step(&mut self, input: &Tensor, index: usize) -> Result<Vec<u32>> {
        let device = input.device();
        let mut draft_tokens = Vec::with_capacity(self.lookahead);
        let mut current_input = input.clone();

        // Phase 1: Draft model generates candidate tokens autoregressively
        for i in 0..self.lookahead {
            let logits = match &mut self.draft_model {
                LoadedModel::Standard(m) => m.forward(&current_input, index + i)?,
                LoadedModel::Quantized(q) => q.forward(&current_input, index + i)?,
            };
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

        // Phase 2: Target model verifies all draft tokens in one forward pass
        let draft_slice: Vec<u32> = draft_tokens.clone();
        let draft_tensor = Tensor::new(draft_slice.as_slice(), device)?.unsqueeze(0)?;
        let target_logits = match &mut self.target_model {
            LoadedModel::Standard(m) => m.forward(&draft_tensor, index)?,
            LoadedModel::Quantized(q) => q.forward(&draft_tensor, index)?,
        };

        // Phase 3: Rejection sampling — accept tokens that match target's top-1
        let mut accepted = Vec::new();
        for i in 0..self.lookahead {
            let target_top = target_logits
                .get(i)?
                .argmax(0)?
                .to_scalar::<u32>()?;
            if target_top == draft_tokens[i] {
                accepted.push(draft_tokens[i]);
            } else {
                accepted.push(target_top);
                break;
            }
        }

        if accepted.is_empty() {
            // Fallback: sample from target at the original position
            let logits = match &mut self.target_model {
                LoadedModel::Standard(m) => m.forward(input, index)?,
                LoadedModel::Quantized(q) => q.forward(input, index)?,
            };
            let token = logits.squeeze(1)?.squeeze(0)?.argmax(0)?.to_scalar::<u32>()?;
            accepted.push(token);
        }

        Ok(accepted)
    }
}
