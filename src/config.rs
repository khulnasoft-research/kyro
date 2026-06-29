use std::path::PathBuf;

use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "kyro",
    version,
    about = "A high-performance ML inference engine"
)]
pub struct Cli {
    /// Path to the model directory or GGUF file
    #[arg(long = "model-path", env = "KYRO_MODEL_PATH")]
    pub model_path: Option<PathBuf>,

    /// Host to bind the API server to
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,

    /// Port to bind the API server to
    #[arg(long, default_value_t = 3000)]
    pub port: u16,

    /// Block size for PagedAttention
    #[arg(long, default_value_t = 16)]
    pub block_size: usize,

    /// Number of GPU blocks
    #[arg(long, default_value_t = 1024)]
    pub num_gpu_blocks: usize,

    /// Number of CPU swap blocks
    #[arg(long, default_value_t = 256)]
    pub num_cpu_blocks: usize,

    /// Maximum tokens per scheduling iteration
    #[arg(long, default_value_t = 2048)]
    pub max_tokens_per_iter: usize,

    /// Chunk size for chunked prefill
    #[arg(long, default_value_t = 512)]
    pub max_prefill_chunk_size: usize,

    /// Request timeout in seconds
    #[arg(long, default_value_t = 300.0)]
    pub request_timeout_secs: f64,

    /// Path to the tokenizer.json file (auto-detected from model-path if not specified)
    #[arg(long = "tokenizer-path", env = "KYRO_TOKENIZER_PATH")]
    pub tokenizer_path: Option<PathBuf>,

    /// Execution mode: eager, full-graph, piecewise, kernel-dispatch
    #[arg(long = "execution-mode", default_value = "eager")]
    pub execution_mode: String,
}

impl Cli {
    pub fn parse_or_default() -> Self {
        Self::parse()
    }
}
