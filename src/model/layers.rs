use candle_core::{Device, Result, Tensor};
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
    inv_freq: Tensor,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, _max_seq_len: usize, device: &Device) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..dim / 2)
            .map(|i| {
                let x = i as f64;
                (1.0 / (10000.0f64.powf(2.0 * x / dim as f64))) as f32
            })
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (dim / 2,), device)?;
        Ok(Self { inv_freq })
    }

    pub fn apply(&self, x: &Tensor, index: usize) -> Result<Tensor> {
        let dims = x.dims();
        if dims.len() < 4 {
            return Ok(x.clone());
        }
        let (_batch, seq_len, _n_heads, head_dim) = (dims[0], dims[1], dims[2], dims[3]);

        let positions: Vec<f32> = (0..seq_len).map(|i| (index + i) as f32).collect();
        let positions = Tensor::from_vec(positions, (seq_len,), x.device())?;

        let freqs = positions
            .unsqueeze(1)?
            .matmul(&self.inv_freq.unsqueeze(0)?)?;
        let half_dim = head_dim / 2;
        let cos = freqs.cos()?.reshape((seq_len, half_dim))?;
        let sin = freqs.sin()?.reshape((seq_len, half_dim))?;
        let cos = Tensor::cat(&[&cos, &cos], 1)?.reshape((1, seq_len, 1, head_dim))?;
        let sin = Tensor::cat(&[&sin, &sin], 1)?.reshape((1, seq_len, 1, head_dim))?;

        let x1 = x.narrow(3, 0, half_dim)?;
        let x2 = x.narrow(3, half_dim, half_dim)?;
        let rotated = Tensor::cat(&[&x2.neg()?, &x1], 3)?;

        x.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarBuilder;

    #[test]
    fn test_rms_norm_forward_shape() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
        let norm = RmsNorm::new(64, 1e-6, vb).unwrap();
        let x = Tensor::ones((2, 4, 64), candle_core::DType::F32, &device).unwrap();
        let out = norm.forward(&x).unwrap();
        assert_eq!(out.dims(), &[2, 4, 64]);
        assert_eq!(out.dtype(), candle_core::DType::F32);
    }

    #[test]
    fn test_rotary_embedding_creation() {
        let device = Device::Cpu;
        let rope = RotaryEmbedding::new(64, 4096, &device).unwrap();
        let x = Tensor::ones((1, 1, 4, 64), candle_core::DType::F32, &device).unwrap();
        let out = rope.apply(&x, 0).unwrap();
        assert_eq!(out.dims(), x.dims());
    }

    #[test]
    fn test_rms_norm_different_dims() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
        let norm = RmsNorm::new(128, 1e-5, vb).unwrap();
        let x = Tensor::ones((1, 128), candle_core::DType::F32, &device).unwrap();
        let out = norm.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 128]);
    }
}
