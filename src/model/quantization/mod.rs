#![allow(dead_code)]

pub mod awq;
pub mod fp8;

use candle_core::{Result, Tensor};

pub trait QuantizedLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
    fn unpack_weights(&self) -> Result<Tensor>;
}
