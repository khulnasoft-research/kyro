use crate::distributed::DistributedContext;
use crate::model::config::{HfModelConfig, LlamaConfig, ModelArchitecture};
use crate::model::kv_cache::CacheContext;
use crate::model::model_registry::ModelInstance;
use crate::model::pipeline::PipelineContext;
use crate::model::quantized::QuantizedLlama;
use anyhow::Context;
use candle_core::{Device, Result, Tensor};
use std::path::Path;

pub trait ModelForward: Send {
    fn forward(
        &mut self,
        x: &Tensor,
        index: usize,
        cache: Option<&mut CacheContext>,
    ) -> Result<Tensor>;
}

pub enum LoadedModel {
    Standard(ModelInstance),
    #[allow(dead_code)]
    Quantized(QuantizedLlama),
}

impl ModelForward for LoadedModel {
    fn forward(
        &mut self,
        x: &Tensor,
        index: usize,
        cache: Option<&mut CacheContext>,
    ) -> Result<Tensor> {
        match self {
            LoadedModel::Standard(m) => m.forward(x, index, cache),
            LoadedModel::Quantized(q) => q.forward(x, index, cache),
        }
    }
}

#[allow(dead_code)]
pub struct ModelLoader {
    pub config: Option<LlamaConfig>,
    pub hf_architecture: ModelArchitecture,
    pub model_path: std::path::PathBuf,
    pub is_gguf: bool,
}

impl ModelLoader {
    #[allow(dead_code)]
    pub fn new<P: AsRef<Path>>(model_path: P) -> anyhow::Result<Self> {
        let model_path = model_path.as_ref().to_path_buf();

        let is_gguf = model_path.extension().is_some_and(|e| e == "gguf")
            || model_path.to_string_lossy().contains(".gguf");

        if is_gguf {
            return Ok(Self {
                config: None,
                hf_architecture: ModelArchitecture::Unknown,
                model_path,
                is_gguf: true,
            });
        }

        let config_path = model_path.join("config.json");
        let hf_config = HfModelConfig::from_file(config_path)?;
        let architecture = hf_config.infer_architecture();
        let config = hf_config.to_llama_config();

        Ok(Self {
            config,
            hf_architecture: architecture,
            model_path,
            is_gguf: false,
        })
    }

    #[allow(dead_code)]
    pub fn architecture(&self) -> ModelArchitecture {
        self.hf_architecture.clone()
    }

    #[allow(dead_code)]
    pub fn detect_tokenizer_path(&self) -> Option<std::path::PathBuf> {
        if !self.model_path.is_dir() {
            return None;
        }

        let candidates = [
            "tokenizer.json",
            "tokenizer_config.json",
            "tokenizer.model",
            "vocab.json",
            "spiece.model",
        ];

        for candidate in candidates {
            let candidate_path = self.model_path.join(candidate);
            if candidate_path.exists() {
                return Some(candidate_path);
            }
        }

        None
    }

    #[allow(dead_code)]
    pub fn load(
        &self,
        device: &Device,
        dist: std::sync::Arc<DistributedContext>,
    ) -> anyhow::Result<LoadedModel> {
        if self.is_gguf {
            let q_model = QuantizedLlama::load_gguf(&self.model_path, device)?;
            return Ok(LoadedModel::Quantized(q_model));
        }

        let config = self
            .config
            .as_ref()
            .context("Model config required for non-GGUF loading")?;

        let mut tensors_files = Vec::new();
        let read_dir = std::fs::read_dir(&self.model_path)?;
        for entry in read_dir {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "safetensors") {
                tensors_files.push(path);
            }
        }

        if tensors_files.is_empty() {
            return Err(anyhow::anyhow!("No .safetensors files found"));
        }

        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &tensors_files,
                candle_core::DType::F16,
                device,
            )?
        };

        let pipeline_ctx = PipelineContext::new(
            dist.rank as usize,
            dist.world_size as usize,
            config.num_hidden_layers,
        );

        let architecture = self.hf_architecture.clone();
        let model = ModelInstance::from_architecture(
            &architecture,
            config,
            vb,
            device,
            dist,
            pipeline_ctx,
        )?;
        Ok(LoadedModel::Standard(model))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn write_dummy_config(dir: &TempDir) {
        let config = serde_json::json!({
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000
        });
        fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
    }

    fn make_loader(dir: &TempDir) -> ModelLoader {
        write_dummy_config(dir);
        ModelLoader::new(dir.path()).unwrap()
    }

    #[test]
    fn test_detect_tokenizer_path_none_for_file() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("model.gguf");
        fs::write(&file_path, b"dummy").unwrap();

        let loader = ModelLoader::new(&file_path).unwrap();
        assert!(loader.detect_tokenizer_path().is_none());
    }

    #[test]
    fn test_detect_tokenizer_path_finds_tokenizer_json() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();

        let loader = make_loader(&dir);
        let detected = loader.detect_tokenizer_path().unwrap();
        assert_eq!(detected.file_name().unwrap(), "tokenizer.json");
    }

    #[test]
    fn test_detect_tokenizer_path_finds_tokenizer_model_fallback() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("tokenizer.model"), b"").unwrap();

        let loader = make_loader(&dir);
        let detected = loader.detect_tokenizer_path().unwrap();
        assert_eq!(detected.file_name().unwrap(), "tokenizer.model");
    }

    #[test]
    fn test_detect_tokenizer_path_returns_none_when_no_candidates() {
        let dir = TempDir::new().unwrap();

        let loader = make_loader(&dir);
        assert!(loader.detect_tokenizer_path().is_none());
    }

    #[test]
    fn test_detect_tokenizer_path_precedence() {
        let dir = TempDir::new().unwrap();
        // Both tokenizer.json and tokenizer.model exist — should prefer tokenizer.json
        fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();
        fs::write(dir.path().join("tokenizer.model"), b"").unwrap();

        let loader = make_loader(&dir);
        let detected = loader.detect_tokenizer_path().unwrap();
        assert_eq!(detected.file_name().unwrap(), "tokenizer.json");
    }

    #[test]
    fn test_gguf_loader_architecture_is_unknown() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("model.gguf");
        fs::write(&file_path, b"dummy").unwrap();

        let loader = ModelLoader::new(&file_path).unwrap();
        assert_eq!(loader.architecture(), ModelArchitecture::Unknown);
        assert!(loader.is_gguf);
    }

    #[test]
    fn test_non_gguf_loader_infers_architecture() {
        let dir = TempDir::new().unwrap();
        let config = serde_json::json!({
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000
        });
        fs::write(dir.path().join("config.json"), config.to_string()).unwrap();

        let loader = ModelLoader::new(dir.path()).unwrap();
        assert_eq!(loader.architecture(), ModelArchitecture::DecoderOnly);
        assert!(!loader.is_gguf);
    }

    #[test]
    fn test_loader_config_fields_mapped() {
        let dir = TempDir::new().unwrap();
        let config = serde_json::json!({
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "vocab_size": 32000,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0
        });
        fs::write(dir.path().join("config.json"), config.to_string()).unwrap();

        let loader = ModelLoader::new(dir.path()).unwrap();
        let cfg = loader.config.as_ref().unwrap();
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.intermediate_size, 8192);
        assert_eq!(cfg.num_hidden_layers, 16);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert!((cfg.rms_norm_eps - 1e-5).abs() < 1e-12);
        assert!((cfg.rope_theta - 500000.0).abs() < f32::EPSILON);
    }
}
