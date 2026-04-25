#![allow(dead_code)]

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module};

#[allow(dead_code)]
pub struct VisionEncoder {
    pub patch_embed: Linear,
    pub layers: Vec<VisionTransformerBlock>,
    pub ln_post: candle_nn::LayerNorm,
}

#[allow(dead_code)]
pub struct VisionTransformerBlock {
    pub ln_1: candle_nn::LayerNorm,
    pub self_attn: candle_nn::Linear,
    pub ln_2: candle_nn::LayerNorm,
    pub mlp: candle_nn::Linear,
}

impl VisionEncoder {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.clone())
    }
}

#[allow(dead_code)]
pub struct VisionLanguageProjection {
    pub linear_1: Linear,
    pub linear_2: Linear,
}

impl VisionLanguageProjection {
    pub fn new(linear_1: Linear, linear_2: Linear) -> Self {
        Self { linear_1, linear_2 }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear_1.forward(x)?;
        let x = x.relu()?;
        self.linear_2.forward(&x)
    }
}
