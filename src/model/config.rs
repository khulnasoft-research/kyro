use anyhow::Result;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize, Clone)]
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
        }
    }
}
