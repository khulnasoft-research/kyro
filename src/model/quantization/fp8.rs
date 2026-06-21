use super::QuantizedLayer;
use candle_core::{DType, Result, Tensor};

#[allow(dead_code)]
pub struct Fp8Linear {
    pub weight: Tensor,
    pub scale: Tensor,
    pub bias: Option<Tensor>,
}

impl Fp8Linear {
    #[allow(dead_code)]
    pub fn new(weight: Tensor, scale: Tensor, bias: Option<Tensor>) -> Self {
        Self {
            weight,
            scale,
            bias,
        }
    }
}

impl QuantizedLayer for Fp8Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_fp8 = x.to_dtype(DType::F8E4M3)?; // Simulation
        let res = x_fp8.matmul(&self.weight)?;

        let res = res.broadcast_mul(&self.scale)?;

        if let Some(bias) = &self.bias {
            res.broadcast_add(bias)
        } else {
            Ok(res)
        }
    }

    fn unpack_weights(&self) -> Result<Tensor> {
        self.weight.to_dtype(DType::F32)
    }
}
