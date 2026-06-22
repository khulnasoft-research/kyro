pub mod awq;
pub mod fp8;
pub mod gptq;
pub mod int4;
pub mod int8;

use candle_core::{DType, Result, Tensor};

pub trait QuantizedLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
    fn unpack_weights(&self) -> Result<Tensor>;
}

pub fn dtype_size(dt: DType) -> usize {
    match dt {
        DType::F32 => 4,
        DType::F16 | DType::BF16 => 2,
        DType::I64 => 8,
        DType::I32 | DType::U32 => 4,
        DType::I16 => 2,
        DType::U8 => 1,
        _ => panic!("unexpected dtype {:?}", dt),
    }
}

/// Pack a slice of 4-bit values into a byte array (2 values per byte).
pub fn pack_i4(values: &[u8]) -> Vec<u8> {
    let packed_len = (values.len() + 1) / 2;
    let mut packed = vec![0u8; packed_len];
    for i in 0..values.len() / 2 {
        packed[i] = (values[i * 2] & 0x0F) | ((values[i * 2 + 1] & 0x0F) << 4);
    }
    if values.len() % 2 == 1 {
        packed[packed_len - 1] = values[values.len() - 1] & 0x0F;
    }
    packed
}

/// Unpack a byte array into 4-bit values.
pub fn unpack_i4(packed: &[u8], num_values: usize) -> Vec<u8> {
    let mut values = Vec::with_capacity(num_values);
    for &byte in packed {
        values.push(byte & 0x0F);
        if values.len() < num_values {
            values.push((byte >> 4) & 0x0F);
        }
    }
    values.truncate(num_values);
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_i4_roundtrip() {
        let original = vec![3, 7, 1, 15, 0, 8, 4, 12];
        let packed = pack_i4(&original);
        let unpacked = unpack_i4(&packed, original.len());
        assert_eq!(original, unpacked);
    }

    #[test]
    fn test_pack_unpack_i4_odd_length() {
        let original = vec![3, 7, 1, 15, 0];
        let packed = pack_i4(&original);
        let unpacked = unpack_i4(&packed, original.len());
        assert_eq!(original, unpacked);
    }

    #[test]
    fn test_dtype_size_f32() {
        assert_eq!(dtype_size(DType::F32), 4);
    }

    #[test]
    fn test_dtype_size_f16() {
        assert_eq!(dtype_size(DType::F16), 2);
    }
}
