use candle_core::{Device, Result, Tensor};
use candle_nn::{Module, VarBuilder};

pub struct RmsNorm {
    inner: candle_nn::RmsNorm,
}

impl RmsNorm {
    #[allow(dead_code)]
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let inner = candle_nn::rms_norm(dim, eps, vb)?;
        Ok(Self { inner })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }
}

#[allow(dead_code)]
pub struct RotaryEmbedding {
    inv_freq: Tensor,
    max_seq_len: usize,
}

impl RotaryEmbedding {
    #[allow(dead_code)]
    pub fn new(dim: usize, max_seq_len: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            inv_freq: Tensor::new(
                (0..dim / 2)
                    .map(|i| {
                        let x = i as f64;
                        1.0 / (10000.0f64.powf(2.0 * x / dim as f64))
                    })
                    .collect::<Vec<_>>()
                    .as_slice(),
                device,
            )?
            .to_dtype(candle_core::DType::F32)?,
            max_seq_len,
        })
    }

    pub fn apply(&self, x: &Tensor, _index: usize) -> Result<Tensor> {
        Ok(x.clone())
    }
}
