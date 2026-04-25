use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module};
use std::collections::HashMap;

pub struct LoraAdapter {
    pub id: String,
    pub a: Tensor, // [rank, hidden_in]
    pub b: Tensor, // [hidden_out, rank]
    pub alpha: f64,
    pub rank: usize,
}

pub struct LoraLinear {
    pub base: Linear,
    pub adapters: HashMap<String, LoraAdapter>,
}

impl LoraLinear {
    pub fn new(base: Linear) -> Self {
        Self {
            base,
            adapters: HashMap::new(),
        }
    }

    pub fn add_adapter(&mut self, adapter: LoraAdapter) {
        self.adapters.insert(adapter.id.clone(), adapter);
    }

    pub fn forward(&self, x: &Tensor, adapter_id: Option<&str>) -> Result<Tensor> {
        let base_out = self.base.forward(x)?;
        
        if let Some(id) = adapter_id {
            if let Some(adapter) = self.adapters.get(id) {
                // lora_out = base_out + (x @ A.T @ B.T) * (alpha / rank)
                let lora_x = x.matmul(&adapter.a.t()?)?;
                let lora_out = lora_x.matmul(&adapter.b.t()?)?;
                let scaling = adapter.alpha / (adapter.rank as f64);
                return base_out.broadcast_add(&(lora_out * scaling)?);
            }
        }
        
        Ok(base_out)
    }
}
