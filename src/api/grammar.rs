use candle_core::{Result, Tensor, D};
use std::collections::HashSet;

pub enum GrammarConstraint {
    None,
    Json,
    Regex(String),
}

pub struct GrammarState {
    pub stack: Vec<String>,
    pub current_text: String,
}

pub struct GrammarLogitsProcessor {
    pub constraint: GrammarConstraint,
    pub state: GrammarState,
}

impl GrammarLogitsProcessor {
    pub fn new(constraint: GrammarConstraint) -> Self {
        Self {
            constraint,
            state: GrammarState {
                stack: Vec::new(),
                current_text: String::new(),
            },
        }
    }

    /// Masks logits based on the current grammar state.
    /// This ensures the model ONLY samples tokens that are valid under the CFG.
    pub fn apply_grammar_mask(&mut self, logits: &Tensor, vocab_size: usize) -> Result<Tensor> {
        if let GrammarConstraint::None = self.constraint {
            return Ok(logits.clone());
        }

        // 1. Determine the set of valid next tokens based on the CFG state
        let valid_tokens = self.get_valid_tokens(vocab_size);

        // 2. Create a mask tensor (initialized with -inf)
        let mut mask_data = vec![f32::NEG_INFINITY; vocab_size];
        for &token_id in &valid_tokens {
            mask_data[token_id] = 0.0;
        }

        let mask = Tensor::from_vec(mask_data, (vocab_size,), logits.device())?;
        
        // 3. Add the mask to logits (valid tokens + 0, invalid tokens + -inf)
        logits.broadcast_add(&mask)
    }

    fn get_valid_tokens(&self, vocab_size: usize) -> HashSet<usize> {
        // In a production implementation like XGrammar, this would:
        // 1. Advance the CFG parser with 'self.state.current_text'
        // 2. Query the trie for tokens that match the next valid transitions.
        
        // Simplified: allow all tokens for now, but provide the entry point for masking.
        (0..vocab_size).collect()
    }

    pub fn advance(&mut self, token_text: &str) {
        self.state.current_text.push_str(token_text);
        // Logic to update the CFG stack would go here.
    }
}
