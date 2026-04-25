#![allow(dead_code)]

use candle_core::{Result, Tensor};
use std::collections::HashSet;

#[derive(Default)]
#[allow(dead_code)]
pub enum GrammarConstraint {
    #[default]
    None,
    Json,
    Regex(String),
}

#[derive(Default)]
#[allow(dead_code)]
pub struct GrammarState {
    pub stack: Vec<String>,
    pub current_text: String,
}

#[allow(dead_code)]
pub struct GrammarLogitsProcessor {
    pub constraint: GrammarConstraint,
    pub state: GrammarState,
}

impl GrammarLogitsProcessor {
    pub fn new(constraint: GrammarConstraint) -> Self {
        Self {
            constraint,
            state: GrammarState::default(),
        }
    }

    pub fn apply_grammar_mask(&mut self, logits: &Tensor, vocab_size: usize) -> Result<Tensor> {
        let valid_tokens = self.get_valid_tokens(vocab_size);
        let mut mask_data = vec![f32::NEG_INFINITY; vocab_size];
        for &token_id in &valid_tokens {
            mask_data[token_id] = 0.0;
        }
        let mask = Tensor::from_slice(&mask_data, (vocab_size,), logits.device())?;
        logits.broadcast_add(&mask)
    }

    fn get_valid_tokens(&self, vocab_size: usize) -> HashSet<usize> {
        match &self.constraint {
            GrammarConstraint::None => (0..vocab_size).collect(),
            GrammarConstraint::Json => {
                let valid = vec![0, 10, 32, 34, 91, 93, 123, 125];
                valid.into_iter().collect()
            }
            GrammarConstraint::Regex(_) => (0..vocab_size).collect(),
        }
    }

    pub fn advance(&mut self, token_text: &str) {
        self.state.current_text.push_str(token_text);
    }
}
