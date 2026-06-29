use std::path::Path;

use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use crate::model::quantization::int4::Int4Linear;

/// Loads a GPTQ-quantized model from safetensors files.
/// GPTQ uses INT4 group-wise quantization with scales and zeros.
#[allow(dead_code)]
pub struct GptqLoader {
    pub model_path: std::path::PathBuf,
    pub group_size: usize,
}

impl GptqLoader {
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

    /// Load quantized weight tensors from safetensors files using VarBuilder.
    #[allow(dead_code)]
    pub fn load_linear(&self, name: &str, device: &Device) -> Result<Int4Linear> {
        let files = self.collect_safetensors()?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::F16, device)? };

        let qweight = vb.get(0, name).or_else(|_| {
            let n = format!("{}.qweight", name);
            vb.get(0, &n)
        })?;
        let scales = {
            let n = format!("{}.scales", name);
            vb.get(0, &n)?
        };
        let qzeros = {
            let n = format!("{}.qzeros", name);
            vb.get(0, &n).ok()
        };

        Ok(Int4Linear::new(
            qweight,
            scales,
            qzeros,
            None,
            self.group_size,
        ))
    }

    fn collect_safetensors(&self) -> Result<Vec<std::path::PathBuf>> {
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

    #[test]
    fn test_gptq_loader_creation() {
        let loader = GptqLoader::new("/tmp/models/gptq-model");
        assert_eq!(loader.group_size, 128);
    }

    #[test]
    fn test_gptq_loader_with_group_size() {
        let loader = GptqLoader::new("/tmp/models/gptq-model").with_group_size(64);
        assert_eq!(loader.group_size, 64);
    }
}
