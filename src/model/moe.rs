use candle_core::{Result, Tensor, D};
use candle_nn::{Linear, Module};

#[allow(dead_code)]
pub struct MoeLayer {
    pub gate: Linear,
    pub experts: Vec<Linear>,
    pub top_k: usize,
}

impl MoeLayer {
    #[allow(dead_code)]
    pub fn new(gate: Linear, experts: Vec<Linear>, top_k: usize) -> Self {
        Self {
            gate,
            experts,
            top_k,
        }
    }

    #[allow(dead_code)]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dims = x.dims();
        let (batch_size, seq_len, hidden_dim) = x.dims3()?;
        let num_tokens = batch_size * seq_len;
        let x_flat = x.reshape((num_tokens, hidden_dim))?;

        // 1. Get routing logits and probabilities
        let gate_logits = self.gate.forward(&x_flat)?;
        let gate_probs = candle_nn::ops::softmax(&gate_logits, D::Minus1)?;

        // 2. Dispatch to experts weighted by routing probabilities
        let mut output = Tensor::zeros_like(&x_flat)?;

        // Simplified approach: for each expert, dispatch tokens with highest routing weight
        for expert_idx in 0..self.experts.len() {
            let expert_weight = gate_probs.get(expert_idx)?.unsqueeze(1)?;
            let expert_output = self.experts[expert_idx].forward(&x_flat)?;
            let contribution = (expert_output * expert_weight)?;
            output = (output + contribution)?;
        }

        output.reshape(orig_dims)
    }
}
