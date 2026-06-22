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
pub struct GrammarState {
    #[allow(dead_code)]
    pub stack: Vec<String>,
    pub current_text: String,
}

pub struct GrammarLogitsProcessor {
    #[allow(dead_code)]
    pub constraint: GrammarConstraint,
    pub state: GrammarState,
}

impl GrammarLogitsProcessor {
    #[allow(dead_code)]
    pub fn new(constraint: GrammarConstraint) -> Self {
        Self {
            constraint,
            state: GrammarState::default(),
        }
    }

    #[allow(dead_code)]
    pub fn apply_grammar_mask(&mut self, logits: &Tensor, vocab_size: usize) -> Result<Tensor> {
        let valid_tokens = self.get_valid_tokens(vocab_size);
        let mut mask_data = vec![f32::NEG_INFINITY; vocab_size];
        for &token_id in &valid_tokens {
            mask_data[token_id] = 0.0;
        }
        let mask = Tensor::from_slice(&mask_data, (vocab_size,), logits.device())?;
        logits.broadcast_add(&mask)
    }

    #[allow(dead_code)]
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

    #[allow(dead_code)]
    pub fn advance(&mut self, token_text: &str) {
        self.state.current_text.push_str(token_text);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_no_constraint_allows_all_tokens() {
        let proc = GrammarLogitsProcessor::new(GrammarConstraint::None);
        let valid = proc.get_valid_tokens(100);
        assert_eq!(valid.len(), 100);
        assert!(valid.contains(&0));
        assert!(valid.contains(&99));
    }

    #[test]
    fn test_json_constraint_limited_tokens() {
        let proc = GrammarLogitsProcessor::new(GrammarConstraint::Json);
        let valid = proc.get_valid_tokens(100);
        // Only 8 specific tokens should be valid
        assert_eq!(valid.len(), 8);
        assert!(valid.contains(&0)); // NUL
        assert!(valid.contains(&10)); // newline
        assert!(valid.contains(&32)); // space
        assert!(valid.contains(&34)); // "
        assert!(valid.contains(&91)); // [
        assert!(valid.contains(&93)); // ]
        assert!(valid.contains(&123)); // {
        assert!(valid.contains(&125)); // }
        assert!(!valid.contains(&1));
    }

    #[test]
    fn test_regex_constraint_passthrough() {
        let proc = GrammarLogitsProcessor::new(GrammarConstraint::Regex(".+".to_string()));
        let valid = proc.get_valid_tokens(50);
        assert_eq!(valid.len(), 50);
    }

    #[test]
    fn test_apply_grammar_mask_none() {
        let device = Device::Cpu;
        let mut proc = GrammarLogitsProcessor::new(GrammarConstraint::None);
        let logits = Tensor::ones((5,), DType::F32, &device).unwrap();
        let masked = proc.apply_grammar_mask(&logits, 5).unwrap();
        // With no constraint, all tokens should remain unchanged (0.0 mask)
        let data: Vec<f32> = masked.flatten_all().unwrap().to_vec1().unwrap();
        assert!(data.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_apply_grammar_mask_json() {
        let device = Device::Cpu;
        let mut proc = GrammarLogitsProcessor::new(GrammarConstraint::Json);
        // Use vocab_size large enough to include all JSON tokens (max index 125)
        let vocab_size = 128;
        let logits = Tensor::ones((vocab_size,), DType::F32, &device).unwrap();
        let masked = proc.apply_grammar_mask(&logits, vocab_size).unwrap();
        let data: Vec<f32> = masked.flatten_all().unwrap().to_vec1().unwrap();
        // Token 0 should remain 1.0 (valid JSON token)
        assert!((data[0] - 1.0).abs() < 1e-6, "token 0 should be valid");
        // Token 1 should be -inf + 1.0 = -inf (not in JSON set)
        assert!(
            data[1].is_infinite() && data[1].is_sign_negative(),
            "token 1 should be masked"
        );
        // Token 125 (}) should be valid
        assert!((data[125] - 1.0).abs() < 1e-6, "token 125 should be valid");
    }

    #[test]
    fn test_advance_appends_text() {
        let mut proc = GrammarLogitsProcessor::new(GrammarConstraint::None);
        assert_eq!(proc.state.current_text, "");
        proc.advance("hello");
        assert_eq!(proc.state.current_text, "hello");
        proc.advance(" world");
        assert_eq!(proc.state.current_text, "hello world");
    }
}
