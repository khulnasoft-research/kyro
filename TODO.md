# Kyro Project TODO

## Current status (based on existing source)

The repository already contains foundational work for:
- Continuous batching scheduler with chunked prefill and request prioritization
- Prefix caching with a radix-style cache in `scheduler/block_manager.rs`
- Speculative decoding scaffolding in `src/speculative.rs`
- GGUF quantized model loading via `candle` in `src/model/quantized.rs`
- Safetensors model loading and standard LLaMA model construction in `src/model/loader.rs`
- OpenAI-compatible `/v1/chat/completions` endpoint in `src/api/openai.rs`
- SSE streaming channel support for token streaming
- Basic device abstraction and CUDA feature gating in `Cargo.toml`
- Distributed context scaffolding in `src/distributed.rs`
- Grammar processor placeholder in scheduler requests
- Prometheus metrics and tracing instrumentation

## TODO: Core inference and throughput

- [ ] Implement state-of-the-art serving throughput measurement and tuning
- [ ] Add end-to-end benchmarks for prefill/decode throughput and latency
- [ ] Optimize scheduler logic for max throughput across batched requests
- [ ] Add chunked prefill full support for large prompts beyond current model stub
- [x] Ensure prefix caching is integrated into actual KV allocation and reuse flow (concat-based KVCacheManager added; radix wiring pending)
- [ ] Add disaggregated prefill / decode / encode pipeline stages
- [ ] Add explicit decode loop separation for prefill-bound vs decode-bound work

## TODO: Model execution and kernel flexibility

- [x] Design a modular execution engine that supports both piecewise and full CUDA/HIP compute graphs
- [x] Add support for full CUDA/HIP graph execution paths for large transformer graphs
- [x] Add piecewise graph execution mode for flexible scheduling and memory reuse
- [x] Create a backend abstraction layer to select between CUDA, HIP, CPU, and other accelerators
- [ ] Implement runtime kernel selection for attention, GEMM, and MoE operations (stub added in KernelDispatch mode)
- [ ] Implement automatic kernel generation and graph-level transformation integration with `torch.compile` or equivalent

## TODO: Quantization and model formats

- [x] Add FP8 quantization support
- [ ] Add MXFP8 / MXFP4 support
- [ ] Add NVFP4 support
- [x] Add INT8 and INT4 model loading and execution paths
- [x] Add GPTQ / AWQ quantization support and loader support
- [ ] Add GGUF support for additional quantized tensor formats and compressed tensors
- [ ] Add support for ModelOpt / TorchAO format loading or conversion integration
- [ ] Add compressed tensor streaming/streamed loading support for very large models

## TODO: Optimized kernels and hardware-specific execution

- [ ] Integrate optimized attention kernels: FlashAttention, FlashInfer, FlashMLA, TRTLLM-GEN, Triton
- [ ] Integrate optimized GEMM/MoE kernels across precision ranges with CUTLASS, TRTLLM-GEN, CuTeDSL
- [ ] Add hardware-specific kernel dispatch for NVIDIA and AMD GPUs
- [ ] Add CPU kernel improvements for x86/ARM/PowerPC
- [ ] Add support for Apple Silicon and other non-NVIDIA accelerators as plugin backends

## TODO: Speculative decoding and decoding algorithms

- [x] Expand speculative decoding support to include n-gram, suffix, EAGLE, DFlash, and other multi-step heuristics
- [x] Add support for parallel sampling / speculative sampling algorithms
- [ ] Add beam search and other high-throughput decoding strategies
- [ ] Add support for constrained and structured decoding with grammar / prefix constraints

## TODO: API, integration, and usability

- [ ] Add seamless Hugging Face model integration and auto-detection for HF format models
- [ ] Support Hugging Face model architectures for:
  - Decoder-only LLMs (e.g., Llama, Qwen, Gemma) (Llama + Qwen2 added)
  - Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
  - Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
  - Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
  - Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
  - Reward and classification models (e.g., Qwen-Math)
- [ ] Add Anthropic Messages API compatibility
- [ ] Add gRPC serving support
- [ ] Add tool calling and reasoning parser infrastructure
- [ ] Add xgrammar / guidance structured output generation support
- [ ] Add streaming output improvements for lower time-to-first-token and smoother token flush
- [x] Add richer OpenAI-compatible options such as `stop`, `logit_bias`, `best_of`, and `top_k`

## TODO: Distributed inference and parallelism

- [x] Expand distributed context beyond single-node placeholder
- [x] Add tensor parallelism support for model weights and attention
- [ ] Add pipeline parallelism support across layers or blocks
- [ ] Add data parallelism support for batching and multi-request scaling
- [ ] Add expert parallelism for Mixture-of-Experts models
- [ ] Add context parallelism for very long sequences and sharded KV cache
- [x] Add multi-node synchronization and NCCL / RCCL integration

## TODO: LoRA and model adaptation

- [ ] Add efficient multi-LoRA support for dense layers
- [ ] Add LoRA support for MoE/expert layers
- [ ] Add runtime merge/unmerge and adapter stacking for multiple LoRAs
- [ ] Add LoRA-aware quantization and execution

## TODO: Hardware plugin ecosystem

- [ ] Define plugin API for GPU/accelerator backends
- [ ] Add NVIDIA GPU support beyond basic CUDA and `cudarc`
- [ ] Add AMD GPU/HIP support
- [ ] Add Google TPU integration
- [ ] Add Intel Gaudi / Habana support
- [ ] Add IBM Spyre support
- [ ] Add Huawei Ascend support
- [ ] Add Rebellions NPU support
- [ ] Add MetaX GPU support
- [ ] Add Apple Silicon / MPS support

## Notes

This TODO list is intended to capture requested high-level capabilities and the current implementation gaps in the existing codebase. The next step is to break these items into implementation issues or project cards with concrete work packages for each major area.

### Implementation status
- AWQ: Properly implemented `AwqLinear` with int32-level 4-bit unpacking, `g_idx` support, and `AwqLoader` for HuggingFace safetensors (2026-06-22)