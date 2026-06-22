use super::{pack_i4, unpack_i4, QuantizedLayer};
use candle_core::{DType, Result, Tensor};

#[allow(dead_code)]
pub struct Int4Linear {
    pub qweight: Tensor,
    pub scales: Tensor,
    pub zeros: Option<Tensor>,
    pub bias: Option<Tensor>,
    pub group_size: usize,
}

impl Int4Linear {
    #[allow(dead_code)]
    pub fn new(
        qweight: Tensor,
        scales: Tensor,
        zeros: Option<Tensor>,
        bias: Option<Tensor>,
        group_size: usize,
    ) -> Self {
        Self {
            qweight,
            scales,
            zeros,
            bias,
            group_size,
        }
    }

    #[allow(dead_code)]
    pub fn quantize(weight: &Tensor, group_size: usize) -> Result<Self> {
        let device = weight.device();
        let dims = weight.dims();
        let out_features = dims[0];
        let in_features = dims[1];

        let weight_data: Vec<f32> = weight.flatten_all()?.to_vec1()?;
        let num_groups = in_features.div_ceil(group_size);
        let mut qweight_data = vec![0u8; (out_features * in_features + 1) / 2];
        let mut scales_data = vec![0.0f32; out_features * num_groups];
        let mut zeros_data = vec![0.0f32; out_features * num_groups];
        let mut values_4bit = vec![0u8; out_features * in_features];

        for row in 0..out_features {
            for g in 0..num_groups {
                let start = g * group_size;
                let end = std::cmp::min(start + group_size, in_features);
                let mut min_val = f32::MAX;
                let mut max_val = f32::MIN;

                for col in start..end {
                    let val = weight_data[row * in_features + col];
                    min_val = min_val.min(val);
                    max_val = max_val.max(val);
                }

                let scale = if max_val - min_val > 1e-10 {
                    (max_val - min_val) / 15.0
                } else {
                    1.0
                };
                let zero_point = (-min_val / scale).round().clamp(0.0, 15.0) as u8;

                scales_data[row * num_groups + g] = scale;
                zeros_data[row * num_groups + g] = zero_point as f32;

                for col in start..end {
                    let val = weight_data[row * in_features + col];
                    let q = ((val / scale) + zero_point as f32).round().clamp(0.0, 15.0) as u8;
                    values_4bit[row * in_features + col] = q;
                }
            }
        }

        qweight_data = pack_i4(&values_4bit);

        let qweight = Tensor::from_slice(&qweight_data, (out_features, (in_features + 1) / 2), device)?;
        let scales = Tensor::from_slice(&scales_data, (out_features, num_groups), device)?;
        let zeros = Tensor::from_slice(&zeros_data, (out_features, num_groups), device)?;

        let device_w = weight.device();
        let qweight = qweight.to_device(device_w)?;
        let scales = scales.to_device(device_w)?;
        let zeros = zeros.to_device(device_w)?;

        Ok(Self {
            qweight,
            scales,
            zeros: Some(zeros),
            bias: None,
            group_size,
        })
    }
}

impl QuantizedLayer for Int4Linear {
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
        let device = self.qweight.device();
        let dims = self.qweight.dims();
        let out_features = dims[0];
        let packed_cols = dims[1];
        let in_features = packed_cols * 2;

        let packed_data: Vec<u8> = self.qweight.flatten_all()?.to_vec1()?;
        let num_values = out_features * in_features;
        let qvalues = unpack_i4(&packed_data, num_values);

        let scales_data: Vec<f32> = self.scales.flatten_all()?.to_vec1()?;
        let num_groups = scales_data.len() / out_features;
        let zeros_data: Vec<f32> = self.zeros.as_ref()
            .map(|z| z.flatten_all().unwrap().to_vec1().unwrap())
            .unwrap_or_else(|| vec![8.0f32; out_features * num_groups]);

        let mut weight_f32 = vec![0.0f32; out_features * in_features];
        for row in 0..out_features {
            for g in 0..num_groups {
                let start = g * self.group_size;
                let end = std::cmp::min(start + self.group_size, in_features);
                let scale = scales_data[row * num_groups + g];
                let zero = zeros_data[row * num_groups + g] as u8;

                for col in start..end {
                    let q = qvalues[row * in_features + col];
                    let deq = (q as f32 - zero as f32) * scale;
                    weight_f32[row * in_features + col] = deq;
                }
            }
        }

        Tensor::from_slice(&weight_f32, (out_features, in_features), device)?.to_dtype(DType::F16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_int4_quantize_roundtrip() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let device = Device::Cpu;
        let weight = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 4],
            &device,
        )?;
        let q = Int4Linear::quantize(&weight, 4)?;
        let reconstructed = q.unpack_weights()?;
        let reconstructed_f32 = reconstructed.to_dtype(DType::F32)?;
        let diff = (weight.clone() - reconstructed_f32)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        // With 4-bit quantization, expect bounded error
        assert!(diff < 20.0, "quantization error too large: {}", diff);
        Ok(())
    }

    #[test]
    fn test_int4_linear_creation() {
        let device = Device::Cpu;
        let weight = Tensor::ones((4, 8), DType::F32, &device).unwrap();
        let q = Int4Linear::quantize(&weight, 4).unwrap();
        assert_eq!(q.group_size, 4);
        assert!(q.scales.dims().len() == 2);
    }
}
