pub mod fp8;
pub mod awq;

use candle_core::{Result, Tensor};

pub trait QuantizedLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
}
