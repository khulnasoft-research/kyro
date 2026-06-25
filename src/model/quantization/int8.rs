use super::QuantizedLayer;
use candle_core::{DType, Result, Tensor};

#[allow(dead_code)]
pub struct Int8Linear {
    pub weight: Tensor,
    pub scale: Tensor,
    pub bias: Option<Tensor>,
}

impl Int8Linear {
    #[allow(dead_code)]
    pub fn new(weight: Tensor, scale: Tensor, bias: Option<Tensor>) -> Self {
        Self {
            weight,
            scale,
            bias,
        }
    }

    /// Per-tensor INT8 quantization (symmetric).
    /// Values are mapped to [0, 255] range with offset 128.
    #[allow(dead_code)]
    pub fn quantize(weight: &Tensor) -> Result<Self> {
        let device = weight.device();
        let dims = weight.dims();

        let weight_data: Vec<f32> = weight.flatten_all()?.to_vec1()?;
        let abs_max = weight_data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

        let scale = if abs_max > 0.0 { abs_max / 127.0 } else { 1.0 };

        let quantized: Vec<u8> = weight_data
            .iter()
            .map(|&v| ((v / scale) + 128.0).round().clamp(0.0, 255.0) as u8)
            .collect();

        let qweight = Tensor::from_slice(&quantized, dims, device)?;
        let scale_t = Tensor::new(&[scale], device)?;

        Ok(Self {
            weight: qweight,
            scale: scale_t,
            bias: None,
        })
    }
}

impl QuantizedLayer for Int8Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let weight_f16 = self.unpack_weights()?;
        let res = x.matmul(&weight_f16.t()?)?;
        if let Some(bias) = &self.bias {
            res.broadcast_add(bias)
        } else {
            Ok(res)
        }
    }

    fn unpack_weights(&self) -> Result<Tensor> {
        let w_f32 = self.weight.to_dtype(DType::F32)?;
        let offset = Tensor::new(&[128.0f32], self.weight.device())?;
        let w_centered = w_f32.broadcast_sub(&offset)?;
        let w_deq = w_centered.broadcast_mul(&self.scale)?;
        w_deq.to_dtype(DType::F16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_int8_quantize_roundtrip() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let device = Device::Cpu;
        let weight = Tensor::from_slice(
            &[1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0],
            &[2, 4],
            &device,
        )?;
        let q = Int8Linear::quantize(&weight)?;
        let reconstructed_f16 = q.unpack_weights()?;
        let reconstructed = reconstructed_f16.to_dtype(DType::F32)?;
        let weight_v: Vec<f32> = weight.flatten_all()?.to_vec1()?;
        let recon_v: Vec<f32> = reconstructed.flatten_all()?.to_vec1()?;
        let total_err: f32 = weight_v
            .iter()
            .zip(recon_v.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            total_err < 2.0,
            "quantization error too large: {}",
            total_err
        );
        Ok(())
    }

    #[test]
    fn test_int8_linear_creation() {
        let device = Device::Cpu;
        let weight = Tensor::ones((3, 6), DType::F32, &device).unwrap();
        let q = Int8Linear::quantize(&weight).unwrap();
        assert_eq!(q.weight.dims(), &[3, 6]);
    }
}
