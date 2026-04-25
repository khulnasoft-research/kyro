use std::path::Path;
use candle_core::{Device, Result};
use candle_nn::VarBuilder;
use crate::model::config::LlamaConfig;
use crate::model::llama::LlamaModel;
use crate::model::quantized::QuantizedLlama;
use crate::distributed::DistributedContext;
use crate::model::pipeline::PipelineContext;
use std::sync::Arc;

pub enum LoadedModel {
    Standard(LlamaModel),
    Quantized(QuantizedLlama),
}

pub struct ModelLoader {
    pub config: Option<LlamaConfig>,
    pub model_path: std::path::PathBuf,
    pub is_gguf: bool,
}

impl ModelLoader {
    pub fn new<P: AsRef<Path>>(model_path: P) -> anyhow::Result<Self> {
        let model_path = model_path.as_ref().to_path_buf();
        
        let is_gguf = model_path.extension().map_or(false, |e| e == "gguf") 
                   || model_path.to_string_lossy().contains(".gguf");

        if is_gguf {
            return Ok(Self { config: None, model_path, is_gguf: true });
        }

        let config_path = model_path.join("config.json");
        let config = LlamaConfig::from_file(config_path)?;
        Ok(Self { config: Some(config), model_path, is_gguf: false })
    }

    pub fn load(&self, device: &Device, dist: Arc<DistributedContext>) -> anyhow::Result<LoadedModel> {
        if self.is_gguf {
            let q_model = QuantizedLlama::load_gguf(&self.model_path, device)?;
            return Ok(LoadedModel::Quantized(q_model));
        }

        let config = self.config.as_ref().unwrap();

        // Find all .safetensors files
        let mut tensors_files = Vec::new();
        for entry in std::fs::read_dir(&self.model_path).map_err(|e| candle_core::Error::wrap(e))? {
            let entry = entry.map_err(|e| candle_core::Error::wrap(e))?;
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "safetensors") {
                tensors_files.push(path);
            }
        }

        if tensors_files.is_empty() {
            return Err(anyhow::anyhow!("No .safetensors files found"));
        }

        // Load using VarBuilder from safetensors
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&tensors_files, candle_core::DType::F16, device)?
        };

        let pipeline_ctx = PipelineContext::new(dist.rank, dist.world_size, config.num_hidden_layers);

        let model = LlamaModel::new(config, vb, device, dist, pipeline_ctx)?;
        Ok(LoadedModel::Standard(model))
    }
}
