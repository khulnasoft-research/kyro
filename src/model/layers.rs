use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Module, VarBuilder};

pub struct RmsNorm {
    inner: candle_nn::RmsNorm,
}

impl RmsNorm {
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

pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, device: &Device) -> Result<Self> {
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / 10000f32.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, device)?.to_dtype(DType::F32)?;
        let freqs = t
            .reshape((max_seq_len, 1))?
            .matmul(&inv_freq.reshape((1, inv_freq.dims()[0]))?)?;
        let sin = freqs.sin()?;
        let cos = freqs.cos()?;
        Ok(Self { sin, cos })
    }

    pub fn apply(&self, x: &Tensor, index: usize) -> Result<Tensor> {
        // x: (batch, seq_len, n_heads, head_dim)
        let (b_sz, seq_len, n_heads, head_dim) = x.dims4()?;
        let cos = self.cos.narrow(0, index, seq_len)?;
        let sin = self.sin.narrow(0, index, seq_len)?;

        // Split x into real and imaginary parts (even and odd indices)
        let x1 = x.narrow(3, 0, head_dim / 2)?;
        let x2 = x.narrow(3, head_dim / 2, head_dim / 2)?;

        // cos and sin are (seq_len, head_dim / 2)
        // Reshape for broadcasting: (1, seq_len, 1, head_dim / 2)
        let cos = cos.reshape((1, seq_len, 1, head_dim / 2))?;
        let sin = sin.reshape((1, seq_len, 1, head_dim / 2))?;

        let out1 = (x1.broadcast_mul(&cos)?).broadcast_sub(&x2.broadcast_mul(&sin)?)?;
        let out2 = (x1.broadcast_mul(&sin)?).broadcast_add(&x2.broadcast_mul(&cos)?)?;

        Tensor::cat(&[out1, out2], 3)
    }
}
