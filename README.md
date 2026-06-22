# Kyro LLM Engine

Kyro is a high-throughput LLM serving engine written in Rust, inspired by vLLM and TGI. It leverages the `candle` ML framework for tensor operations and `tokio` for high-concurrency async scheduling.

## Status

Kyro is under active development. The core architecture (continuous batching, PagedAttention, prefix caching, speculative decoding) is functional with a mock model for testing. Real model loading (GGUF, Safetensors) is supported via the `--model-path` CLI argument.

## Key Features

- **Continuous Batching**: Iteration-level scheduling to maximize GPU throughput.
- **PagedAttention**: Block-sparse KV cache management with virtual memory indirection.
- **Prefix Caching (Radix Cache)**: Automatic reuse of KV cache for common prefixes via longest-prefix matching and LRU eviction.
- **Chunked Prefill**: Interleaves large prompt processing with active decode steps to avoid "prefill stall".
- **Speculative Decoding**: Rejection-sampling based verification of draft tokens against the target model.
- **Quantization Support**: GGUF format weight loading via `candle`.
- **Distributed Inference**: Basic Tensor/Pipeline Parallelism scaffolding with `All-Reduce`.
- **Constrained Decoding**: Grammar-based logit masking for JSON-mode and regex-constrained output.
- **OpenAI-compatible API**: `/v1/chat/completions` with streaming SSE support, tokenization, and usage statistics.
- **Observability**: Prometheus metrics (`/metrics`) and structured tracing.

## Architecture

```
Request → API (Axum) → Scheduler (Continuous Batching) → Worker (Model Inference)
                             ↕                                      ↕
                        Block Manager ←─────────────────── PagedAttention Kernel
                             ↕
                       Radix Cache (LRU)
```

1. **API Layer** (`src/api/`): Axum-based HTTP handlers for `/v1/chat/completions`, `/health`, `/metrics`. Includes tokenizer (HuggingFace `tokenizers`) and grammar processor.
2. **Scheduler** (`src/scheduler/`): Manages request queues (waiting/running), chunked prefill scheduling, block allocation via `BlockManager`, and prefix caching via `RadixCache`.
3. **Worker** (`src/worker.rs`): Async loop that schedules → runs model inference → updates state. Compute happens outside scheduler lock.
4. **Model** (`src/model/`): LLaMA transformer with PagedAttention kernel, MoE expert dispatch, GGUF quantized loading.
5. **KV Cache** (`src/model/attention_kernel.rs`): Software PagedAttention with block-sparse key/value lookup via block tables.
6. **Distributed** (`src/distributed.rs`): NCCL-based all-reduce for tensor/pipeline parallelism.

## Getting Started

### Prerequisites

- Rust nightly (2026+)
- CUDA 12.x (GPU) or Apple Metal (macOS) or CPU-only

### Quick Start

```bash
# Clone and build
git clone https://github.com/nrelab/kyro.git
cd kyro
cargo build --release

# Run with mock model (no model files needed)
cargo run --release

# Run with a real model
cargo run --release -- --model-path /path/to/llama/model
```

The API will be available at `http://localhost:3000`.

### CLI Arguments

| Argument | Env Variable | Default | Description |
|---|---|---|---|
| `--model-path` | `KYRO_MODEL_PATH` | — | Path to model directory (Safetensors) or GGUF file |
| `--tokenizer-path` | `KYRO_TOKENIZER_PATH` | auto | Path to `tokenizer.json` (auto-detected from model path) |
| `--host` | — | `0.0.0.0` | API bind host |
| `--port` | — | `3000` | API bind port |
| `--block-size` | — | `16` | PagedAttention block size (tokens) |
| `--num-gpu-blocks` | — | `1024` | Number of GPU KV cache blocks |
| `--num-cpu-blocks` | — | `256` | Number of CPU swap blocks |
| `--max-tokens-per-iter` | — | `2048` | Max tokens per scheduling iteration |
| `--max-prefill-chunk-size` | — | `512` | Chunk size for chunked prefill |
| `--request-timeout-secs` | — | `300` | Request timeout in seconds |

### Docker

```bash
# Build image
docker build -t kyro:latest .

# Run with GPU support
docker run --gpus all -p 3000:3000 -v /path/to/models:/models kyro:latest \
  --model-path /models/llama
```

See `docker-compose.yml` and `k8s/` for deployment examples.

## API Documentation

### `POST /v1/chat/completions`

OpenAI-compatible chat completions endpoint. Supports both streaming (SSE) and non-streaming responses.

**Request:**

```json
{
  "model": "kyro-llama",
  "messages": [
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "stream": false,
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9
}
```

**Response (non-streaming):**

```json
{
  "id": "chatcmpl-123456",
  "object": "chat.completion",
  "created": 1719000000,
  "model": "kyro-llama",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "I'm doing well, thank you!"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 6,
    "total_tokens": 16
  }
}
```

**Streaming (SSE):**

```
data: {"id":"chatcmpl-123456","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"I'm"},"finish_reason":null}]}
data: {"id":"chatcmpl-123456","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" doing"},"finish_reason":null}]}
data: [DONE]
```

### `GET /health`

Liveness and readiness probe. Returns `200 OK` when the engine is running.

### `GET /metrics`

Prometheus-formatted metrics:

| Metric | Type | Description |
|---|---|---|
| `kyro_requests_total` | Counter | Total requests processed |
| `kyro_tokens_total` | Counter | Total tokens generated |
| `kyro_token_latency_seconds` | Histogram | Per-token generation latency |
| `kyro_ttft_ms` | Histogram | Time to first token |
| `kyro_tbt_ms` | Histogram | Time between tokens |
| `kyro_kv_cache_usage_percent` | Gauge | KV cache utilization |

### Error Responses

Errors follow OpenAI's format:

```json
{
  "error": {
    "message": "messages must not be empty",
    "type": "invalid_request_error",
    "code": "empty_messages"
  }
}
```

| Status | Error Type | Description |
|---|---|---|
| 400 | `invalid_request_error` | Validation errors (missing fields, invalid values) |
| 422 | `invalid_request_error` | Processing errors (tokenization failure) |
| 500 | `server_error` | Internal engine errors |

## Development

### Running Tests

```bash
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

### Project Layout

```
src/
├── api/
│   ├── openai.rs      # HTTP handlers, request/response types
│   ├── tokenizer.rs   # HuggingFace tokenizer wrapper
│   └── grammar.rs     # Grammar-constrained decoding
├── model/
│   ├── llama.rs       # LLaMA transformer implementation
│   ├── attention_kernel.rs  # PagedAttention kernel
│   ├── config.rs      # Model configuration
│   ├── loader.rs      # Model loading (Safetensors, GGUF)
│   ├── moe.rs         # Mixture-of-Experts routing
│   └── quantized.rs   # GGUF quantized model loading
├── scheduler/
│   ├── continuous_batching.rs  # Scheduler, request management
│   ├── block_manager.rs       # Physical block allocation
│   └── radix_cache.rs         # Prefix caching with LRU eviction
├── main.rs            # Entry point, CLI parsing, wiring
├── config.rs          # CLI argument definitions (Clap)
├── worker.rs          # Inference loop (schedule → compute → update)
├── metrics.rs         # Prometheus metric definitions
├── device.rs          # Hardware detection (CUDA, Metal, CPU)
├── distributed.rs     # Tensor/Pipeline parallelism
└── speculative.rs     # Speculative decoding
```

## License

Apache 2.0. See [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributions must be signed off (DCO).
