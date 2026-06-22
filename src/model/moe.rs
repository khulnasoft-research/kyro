use candle_core::{Result, Tensor, D};
use candle_nn::{Linear, Module};

#[allow(dead_code)]
pub struct MoeLayer {
    pub gate: Linear,
    pub experts: Vec<Linear>,
    pub top_k: usize,
}

#[allow(dead_code)]
impl MoeLayer {
    pub fn new(gate: Linear, experts: Vec<Linear>, top_k: usize) -> Self {
        Self {
            gate,
            experts,
            top_k,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dims = x.dims();
        let (batch_size, seq_len, hidden_dim) = x.dims3()?;
        let num_tokens = batch_size * seq_len;
        let x_flat = x.reshape((num_tokens, hidden_dim))?;

        let gate_logits = self.gate.forward(&x_flat)?;
        let gate_probs = candle_nn::ops::softmax(&gate_logits, D::Minus1)?;

        let num_experts = self.experts.len();
        let (_sorted_vals, sorted_idx) = gate_probs.sort_last_dim(false)?;
        let top_idx = sorted_idx.narrow(1, 0, self.top_k)?;

        let mut output = Tensor::zeros_like(&x_flat)?;

        for expert_idx in 0..num_experts {
            let mask = top_idx.eq(expert_idx as u32)?;
            let in_topk = mask.sum(1)?.to_dtype(x.dtype())?;
            let count = in_topk.sum_all()?.to_scalar::<f64>()?;
            if count < 0.5 {
                continue;
            }
            let weight = gate_probs.narrow(1, expert_idx, 1)?;
            let expert_output = self.experts[expert_idx].forward(&x_flat)?;
            let contribution = (expert_output * weight.broadcast_mul(&in_topk.unsqueeze(1)?)?)?;
            output = (output + contribution)?;
        }

        output.reshape(orig_dims)
    }
}
