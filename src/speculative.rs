#![allow(dead_code)]

use crate::model::loader::LoadedModel;
use candle_core::{Result, Tensor};

pub struct SpeculativeDecoder {
    pub target_model: LoadedModel,
    pub draft_model: LoadedModel,
    pub lookahead: usize,
}

impl SpeculativeDecoder {
    pub fn new(target_model: LoadedModel, draft_model: LoadedModel, lookahead: usize) -> Self {
        Self {
            target_model,
            draft_model,
            lookahead,
        }
    }

    pub fn step(&mut self, input: &Tensor, index: usize) -> Result<Vec<u32>> {
        let device = input.device();
        let mut draft_tokens = Vec::with_capacity(self.lookahead);
        let mut current_input = input.clone();

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

        let _full_spec_seq: Vec<u32> = Vec::new();

        Ok(draft_tokens)
    }
}
