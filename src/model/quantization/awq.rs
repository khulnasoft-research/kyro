use anyhow::Result as AResult;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use std::path::Path;

use super::QuantizedLayer;

#[allow(dead_code)]
pub struct AwqLinear {
    pub qweight: Tensor,
    pub qzeros: Tensor,
    pub scales: Tensor,
    pub g_idx: Option<Tensor>,
    pub bias: Option<Tensor>,
    pub group_size: usize,
}

impl AwqLinear {
    #[allow(dead_code)]
    pub fn new(
        qweight: Tensor,
        qzeros: Tensor,
        scales: Tensor,
        g_idx: Option<Tensor>,
        bias: Option<Tensor>,
        group_size: usize,
    ) -> Self {
        Self {
            qweight,
            qzeros,
            scales,
            g_idx,
            bias,
            group_size,
        }
    }
}

/// Unpack a single i32 word containing 8 consecutive 4-bit values.
#[allow(dead_code)]
fn unpack_i32_word(word: i32) -> [u8; 8] {
    let w = word as u32;
    [
        (w & 0x0F) as u8,
        ((w >> 4) & 0x0F) as u8,
        ((w >> 8) & 0x0F) as u8,
        ((w >> 12) & 0x0F) as u8,
        ((w >> 16) & 0x0F) as u8,
        ((w >> 20) & 0x0F) as u8,
        ((w >> 24) & 0x0F) as u8,
        ((w >> 28) & 0x0F) as u8,
    ]
}

impl QuantizedLayer for AwqLinear {
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
        let weight_shape = self.qweight.dims();
        let out_features = weight_shape[0];
        let in_features = weight_shape[1] * 8;

        let qweight_data: Vec<i32> = self.qweight.flatten_all()?.to_vec1()?;
        let scales_data: Vec<f32> = self.scales.flatten_all()?.to_vec1()?;
        let num_groups = scales_data.len() / out_features;

        // Unpack qzeros (same i32 packing as qweight)
        let qzeros_data: Vec<i32> = if self.qzeros.elem_count() > 0 {
            self.qzeros.flatten_all()?.to_vec1()?
        } else {
            vec![]
        };
        let qzeros_row_stride = if qzeros_data.is_empty() {
            0
        } else {
            qzeros_data.len() / out_features
        };

        // Unpack g_idx if present
        let g_idx_data: Option<Vec<i32>> = self
            .g_idx
            .as_ref()
            .map(|g| g.flatten_all().unwrap().to_vec1().unwrap());

        let mut weight_f32 = vec![0.0f32; out_features * in_features];

        for row in 0..out_features {
            for col in 0..in_features {
                let word_idx = col / 8;
                let nibble = col % 8;
                let word = qweight_data[row * weight_shape[1] + word_idx];
                let q_val = unpack_i32_word(word)[nibble];

                let group = g_idx_data
                    .as_ref()
                    .map(|g| g[col] as usize)
                    .unwrap_or(col / self.group_size);

                let z = if !qzeros_data.is_empty() {
                    let zero_word_idx = group / 8;
                    let zero_nibble = group % 8;
                    let zword = qzeros_data[row * qzeros_row_stride + zero_word_idx];
                    unpack_i32_word(zword)[zero_nibble]
                } else {
                    8u8
                };

                let s = scales_data[row * num_groups + group];
                weight_f32[row * in_features + col] = (q_val as f32 - z as f32) * s;
            }
        }

        let w = Tensor::from_slice(&weight_f32, (out_features, in_features), device)?;
        w.to_dtype(DType::F16)
    }
}

/// Loader for HuggingFace AWQ quantized models.
#[allow(dead_code)]
pub struct AwqLoader {
    pub model_path: std::path::PathBuf,
    pub group_size: usize,
}

impl AwqLoader {
    #[allow(dead_code)]
    pub fn new<P: AsRef<Path>>(model_path: P) -> Self {
        Self {
            model_path: model_path.as_ref().to_path_buf(),
            group_size: 128,
        }
    }

    #[allow(dead_code)]
    pub fn with_group_size(mut self, group_size: usize) -> Self {
        self.group_size = group_size;
        self
    }

    #[allow(dead_code)]
    pub fn load_linear(&self, name: &str, device: &Device) -> AResult<AwqLinear> {
        let files = self.collect_safetensors()?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::F32, device)? };

        let qweight = vb.get(0, &format!("{}.qweight", name))?;
        let qzeros = vb.get(0, &format!("{}.qzeros", name))?;
        let scales = vb.get(0, &format!("{}.scales", name))?;
        let g_idx = vb.get(0, &format!("{}.g_idx", name)).ok();

        Ok(AwqLinear::new(
            qweight,
            qzeros,
            scales,
            g_idx,
            None,
            self.group_size,
        ))
    }

    fn collect_safetensors(&self) -> AResult<Vec<std::path::PathBuf>> {
        let mut files = Vec::new();
        if self.model_path.is_dir() {
            let read_dir = std::fs::read_dir(&self.model_path)?;
            for entry in read_dir {
                let entry = entry?;
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "safetensors") {
                    files.push(path);
                }
            }
        } else if self
            .model_path
            .extension()
            .is_some_and(|ext| ext == "safetensors")
        {
            files.push(self.model_path.clone());
        } else {
            return Err(anyhow::anyhow!(
                "No .safetensors files found at {:?}",
                self.model_path
            ));
        }
        Ok(files)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_unpack_i32_word() {
        let mut word: i32 = 0;
        for i in 0..8i32 {
            word |= i << (i * 4);
        }
        let vals = unpack_i32_word(word);
        assert_eq!(vals, [0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_awq_linear_creation() {
        let device = Device::Cpu;
        let out = 4;
        let inp = 16;
        let qweight = Tensor::zeros((out, inp / 8), DType::I32, &device).unwrap();
        let num_groups = inp / 128 + 1;
        let qzeros = Tensor::zeros((out, num_groups.div_ceil(8)), DType::I32, &device).unwrap();
        let scales = Tensor::ones((out, num_groups), DType::F32, &device).unwrap();
        let awq = AwqLinear::new(qweight, qzeros, scales, None, None, 128);
        assert_eq!(awq.qweight.dims(), &[4, 2]);
    }

    #[test]
    fn test_awq_loader_creation() {
        let loader = AwqLoader::new("/tmp/models/awq-model");
        assert_eq!(loader.group_size, 128);
    }

    #[test]
    fn test_awq_loader_with_group_size() {
        let loader = AwqLoader::new("/tmp/models/awq-model").with_group_size(64);
        assert_eq!(loader.group_size, 64);
    }
}
