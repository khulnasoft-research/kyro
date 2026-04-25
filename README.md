# Kyro LLM Engine

Kyro is a high-throughput LLM serving engine written in Rust, inspired by vLLM and TGI. It leverages the `candle` ML framework for efficient tensor operations and `tokio` for high-concurrency async scheduling.

## Key Features

- **Continuous Batching**: Iteration-level scheduling to maximize GPU throughput and eliminate queue wait times.
- **PagedAttention**: Virtual memory management for KV cache, eliminating memory fragmentation and enabling long-context serving.
- **Prefix Caching (Radix Cache)**: Automatic reuse of KV cache for common prefixes (system prompts, multi-turn history), enabling near-zero Time-To-First-Token (TTFT).
- **Chunked Prefill**: Eliminates "Prefill Stall" by interleaving large prompt processing with active decode steps.
- **Speculative Decoding**: Accelerates generation by 2x using a lightweight draft model for token prediction and a target model for parallel verification.
- **Distributed Inference**: Support for **Tensor Parallelism (TP)** and **Pipeline Parallelism (PP)** to serve massive models across multiple GPUs.
- **Quantization Support**: Native support for **FP8 (Hopper)**, **AWQ (4-bit)**, and **GGUF** weight loading.
- **Constrained Decoding**: Structured JSON-mode and Regex-constrained output via grammar-based sampling.
- **Multi-LoRA Support**: Dynamic loading and switching of many task-specific adapters on a single base model.
- **Observability**: Real-time Prometheus metrics for TTFT, TBT (Time Between Tokens), and KV cache utilization.

## Architecture

1. **Frontend (Axum)**: Handles HTTP requests, streaming SSE, and health/metrics endpoints.
2. **Scheduler (Continuous Batching)**: Manages request queues, prefix caching, and chunked prefill scheduling.
3. **Model (Candle)**: Optimized Transformer blocks with support for multiple quantization formats (FP8, AWQ, GGUF), PagedAttention kernels, and LoRA adapters.
4. **KV Cache (PagedAttention)**: Manages logical-to-physical block mapping via a **Reference-Counted BlockManager**, ensuring cached prefixes are protected from overwrite.
5. **Distributed (NCCL)**: Handles multi-node/multi-GPU synchronization via `All-Reduce`.

## Getting Started

### Running the Engine

```bash
cargo run --release
```

The API will be available at `http://localhost:3000/v1/chat/completions`.

### Benchmarking

To stress test the engine under concurrent load:

```bash
python benchmarks/stress_test.py
```

## API Documentation

- **POST** `/v1/chat/completions`: OpenAI-compatible completions endpoint.
- **GET** `/health`: Liveness and readiness probe.
- **GET** `/metrics`: Prometheus-formatted engine metrics.
