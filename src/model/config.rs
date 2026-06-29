use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct LlamaConfig {
    #[allow(dead_code)]
    pub hidden_size: usize,
    #[allow(dead_code)]
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    #[allow(dead_code)]
    pub num_attention_heads: usize,
    #[allow(dead_code)]
    pub num_key_value_heads: usize,
    #[allow(dead_code)]
    pub vocab_size: usize,
    #[allow(dead_code)]
    pub rms_norm_eps: f64,
    #[allow(dead_code)]
    pub rope_theta: f32,
    #[allow(dead_code)]
    pub max_seq_len: Option<usize>,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelArchitecture {
    DecoderOnly,
    MixtureOfExperts,
    HybridAttentionStateSpace,
    MultiModal,
    EmbeddingRetrieval,
    RewardClassification,
    Unknown,
}

#[derive(Debug, Deserialize)]
pub struct HfModelConfig {
    pub model_type: Option<String>,
    pub architectures: Option<Vec<String>>,
    pub hidden_size: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub num_hidden_layers: Option<usize>,
    pub num_attention_heads: Option<usize>,
    pub num_key_value_heads: Option<usize>,
    pub vocab_size: Option<usize>,
    pub rms_norm_eps: Option<f64>,
    pub rope_theta: Option<f32>,
    pub max_position_embeddings: Option<usize>,
}

impl LlamaConfig {
    #[allow(dead_code)]
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let config: Self = serde_json::from_reader(file)?;
        Ok(config)
    }

    pub fn llama_7b() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            vocab_size: 32000,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            max_seq_len: Some(4096),
        }
    }
}

impl HfModelConfig {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let config: Self = serde_json::from_reader(file)?;
        Ok(config)
    }

    pub fn infer_architecture(&self) -> ModelArchitecture {
        let mut values = Vec::new();
        if let Some(model_type) = &self.model_type {
            values.push(model_type.to_lowercase());
        }
        if let Some(architectures) = &self.architectures {
            for architecture in architectures {
                values.push(architecture.to_lowercase());
            }
        }

        let contains_any = |keywords: &[&str]| {
            values
                .iter()
                .any(|value| keywords.iter().any(|keyword| value.contains(keyword)))
        };

        if contains_any(&["llama", "qwen", "gemma", "gpt", "bloom", "gptj"]) {
            return ModelArchitecture::DecoderOnly;
        }

        if contains_any(&["moe", "mixtral", "deepseek", "gpt-oss"]) {
            return ModelArchitecture::MixtureOfExperts;
        }

        if contains_any(&["mamba", "qwen3.5", "state_space", "ssm", "hybrid", "s4"]) {
            return ModelArchitecture::HybridAttentionStateSpace;
        }

        if contains_any(&[
            "llava",
            "qwen-vl",
            "pixtral",
            "vision",
            "multimodal",
            "vision-language",
        ]) {
            return ModelArchitecture::MultiModal;
        }

        if contains_any(&[
            "e5",
            "mistral",
            "gte",
            "colbert",
            "embedding",
            "retrieval",
            "sbert",
        ]) {
            return ModelArchitecture::EmbeddingRetrieval;
        }

        if contains_any(&["reward", "math", "classification", "qa", "scoring"]) {
            return ModelArchitecture::RewardClassification;
        }

        ModelArchitecture::Unknown
    }

    pub fn to_llama_config(&self) -> Option<LlamaConfig> {
        Some(LlamaConfig {
            hidden_size: self.hidden_size?,
            intermediate_size: self.intermediate_size?,
            num_hidden_layers: self.num_hidden_layers?,
            num_attention_heads: self.num_attention_heads?,
            num_key_value_heads: self
                .num_key_value_heads
                .unwrap_or(self.num_attention_heads?),
            vocab_size: self.vocab_size?,
            rms_norm_eps: self.rms_norm_eps.unwrap_or(1e-6),
            rope_theta: self.rope_theta.unwrap_or(10000.0),
            max_seq_len: self.max_position_embeddings,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama_7b_preset_values() {
        let cfg = LlamaConfig::llama_7b();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.intermediate_size, 11008);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 32);
        assert_eq!(cfg.vocab_size, 32000);
        assert!((cfg.rms_norm_eps - 1e-6).abs() < 1e-12);
    }

    #[test]
    fn test_llama_7b_uses_mha() {
        let cfg = LlamaConfig::llama_7b();
        assert_eq!(cfg.num_attention_heads, cfg.num_key_value_heads);
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let cfg = LlamaConfig::llama_7b();
        let json = serde_json::to_string(&cfg).unwrap();
        let deserialized: LlamaConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.hidden_size, cfg.hidden_size);
        assert_eq!(deserialized.num_hidden_layers, cfg.num_hidden_layers);
    }

    // ── HfModelConfig::infer_architecture tests ──────────────────

    fn hf(model_type: &str, architectures: Vec<&str>) -> HfModelConfig {
        HfModelConfig {
            model_type: Some(model_type.to_string()),
            architectures: Some(architectures.iter().map(|s| s.to_string()).collect()),
            hidden_size: Some(4096),
            intermediate_size: Some(11008),
            num_hidden_layers: Some(32),
            num_attention_heads: Some(32),
            num_key_value_heads: Some(32),
            vocab_size: Some(32000),
            rms_norm_eps: Some(1e-6),
            rope_theta: Some(10000.0),
            max_position_embeddings: Some(4096),
        }
    }

    #[test]
    fn test_infer_decoder_only_llama() {
        let cfg = hf("llama", vec!["LlamaForCausalLM"]);
        assert_eq!(cfg.infer_architecture(), ModelArchitecture::DecoderOnly);
    }

    #[test]
    fn test_infer_decoder_only_qwen() {
        let cfg = hf("qwen2", vec!["QwenForCausalLM"]);
        assert_eq!(cfg.infer_architecture(), ModelArchitecture::DecoderOnly);
    }

    #[test]
    fn test_infer_decoder_only_gemma() {
        let cfg = hf("gemma", vec!["GemmaForCausalLM"]);
        assert_eq!(cfg.infer_architecture(), ModelArchitecture::DecoderOnly);
    }

    #[test]
    fn test_infer_moe_mixtral() {
        let cfg = hf("mixtral", vec!["MixtralForCausalLM"]);
        assert_eq!(
            cfg.infer_architecture(),
            ModelArchitecture::MixtureOfExperts
        );
    }

    #[test]
    fn test_infer_moe_deepseek() {
        let cfg = hf("deepseek", vec!["DeepseekForCausalLM"]);
        assert_eq!(
            cfg.infer_architecture(),
            ModelArchitecture::MixtureOfExperts
        );
    }

    #[test]
    fn test_infer_hybrid_mamba() {
        let cfg = hf("mamba", vec!["MambaForCausalLM"]);
        assert_eq!(
            cfg.infer_architecture(),
            ModelArchitecture::HybridAttentionStateSpace
        );
    }

    #[test]
    fn test_infer_multimodal_llava() {
        let cfg = hf("llava", vec!["LlavaForConditionalGeneration"]);
        assert_eq!(cfg.infer_architecture(), ModelArchitecture::MultiModal);
    }

    #[test]
    fn test_infer_embedding_e5() {
        let cfg = hf("e5", vec!["E5Model"]);
        assert_eq!(
            cfg.infer_architecture(),
            ModelArchitecture::EmbeddingRetrieval
        );
    }

    #[test]
    fn test_infer_unknown() {
        let cfg = hf("some_random_model", vec!["CustomModel"]);
        assert_eq!(cfg.infer_architecture(), ModelArchitecture::Unknown);
    }

    #[test]
    fn test_infer_uses_model_type_when_architectures_missing() {
        let cfg = HfModelConfig {
            model_type: Some("llama".to_string()),
            architectures: None,
            hidden_size: Some(4096),
            intermediate_size: Some(11008),
            num_hidden_layers: Some(32),
            num_attention_heads: Some(32),
            num_key_value_heads: Some(32),
            vocab_size: Some(32000),
            rms_norm_eps: Some(1e-6),
            rope_theta: Some(10000.0),
            max_position_embeddings: Some(4096),
        };
        assert_eq!(cfg.infer_architecture(), ModelArchitecture::DecoderOnly);
    }

    // ── HfModelConfig::to_llama_config tests ─────────────────────

    #[test]
    fn test_to_llama_config_full() {
        let cfg = hf("llama", vec!["LlamaForCausalLM"]);
        let llama = cfg.to_llama_config().unwrap();
        assert_eq!(llama.hidden_size, 4096);
        assert_eq!(llama.num_hidden_layers, 32);
        assert_eq!(llama.num_key_value_heads, 32);
    }

    #[test]
    fn test_to_llama_config_defaults_kv_heads() {
        let cfg = HfModelConfig {
            num_key_value_heads: None,
            ..hf("llama", vec!["LlamaForCausalLM"])
        };
        let llama = cfg.to_llama_config().unwrap();
        assert_eq!(llama.num_key_value_heads, 32); // falls back to num_attention_heads
    }

    #[test]
    fn test_to_llama_config_defaults_eps() {
        let cfg = HfModelConfig {
            rms_norm_eps: None,
            ..hf("llama", vec!["LlamaForCausalLM"])
        };
        let llama = cfg.to_llama_config().unwrap();
        assert!((llama.rms_norm_eps - 1e-6).abs() < 1e-12);
    }

    #[test]
    fn test_to_llama_config_missing_required_field() {
        let cfg = HfModelConfig {
            hidden_size: None,
            ..hf("llama", vec!["LlamaForCausalLM"])
        };
        assert!(cfg.to_llama_config().is_none());
    }
}
