use candle_core::{Result, Tensor, D};
use candle_nn::{Linear, Module};

pub struct MoeLayer {
    pub gate: Linear,
    pub experts: Vec<Linear>, // Simplified: each expert is a Linear layer
    pub top_k: usize,
}

impl MoeLayer {
    pub fn new(gate: Linear, experts: Vec<Linear>, top_k: usize) -> Self {
        Self {
            gate,
            experts,
            top_k,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_dim) = x.dims3()?;
        let x_flat = x.flatten(0, 1)?; // [batch * seq, hidden]

        // 1. Get routing weights
        let gate_logits = self.gate.forward(&x_flat)?;
        let gate_probs = candle_nn::ops::softmax(&gate_logits, D::Minus1)?;

        // 2. Select top-k experts - for simplicity, just take argmax
        let top_k_indices = gate_probs.argmax(D::Minus1)?;

        // 3. Dispatch to experts
        let mut final_output = Tensor::zeros_like(&x_flat)?;

        // Simplified MoE: just use the top expert
        let top_expert_idx = top_k_indices.get(0)?.to_scalar::<i64>()? as usize;
        if top_expert_idx < self.experts.len() {
            final_output = self.experts[top_expert_idx].forward(&x_flat)?;
        }

        final_output.reshape((batch_size, seq_len, hidden_dim))
    }
}
