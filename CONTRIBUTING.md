# Contributing to Kyro

Thank you for your interest in Kyro! We welcome contributions of all kinds — bug reports, feature requests, documentation, and code.

## Code of Conduct

All participants in the Kyro community are expected to adhere to the [Contributor Covenant](CODE_OF_CONDUCT.md). Please read it before engaging.

## Getting Started

### Prerequisites

- Rust nightly (2026+ toolchain)
- CUDA 12.x toolkit (for GPU support) or Apple Metal (macOS)
- `cmake`, `pkg-config`, `libssl-dev`

### Development Setup

```bash
git clone https://github.com/nrelab/kyro.git
cd kyro
cargo build
cargo test
cargo clippy --all-targets
```

### Project Structure

```
src/
├── api/          # HTTP handlers (Axum), tokenizer, grammar
├── model/        # Transformer blocks, PagedAttention, quantized models
├── scheduler/    # Continuous batching, block manager, radix cache
├── worker.rs     # Main inference loop
├── main.rs       # Entry point, wiring
├── config.rs     # CLI argument parsing (Clap)
├── device.rs     # Hardware detection (CUDA/Metal/CPU)
├── metrics.rs    # Prometheus metric definitions
├── distributed.rs# Tensor/pipeline parallelism
└── speculative.rs# Speculative decoding
```

## How to Contribute

### Reporting Bugs

1. Search existing [issues](https://github.com/nrelab/kyro/issues) first.
2. Use the **Bug Report** template when filing.
3. Include: minimal reproduction, expected vs actual behavior, environment details.

### Feature Requests

1. Open an issue using the **Feature Request** template.
2. Describe the use case, expected API, and any prior art.
3. Wait for maintainer feedback before starting implementation.

### Pull Requests

1. **Keep PRs focused** on a single change. Split large features.
2. **Run all checks** before submitting:
   ```bash
   cargo check --all-targets
   cargo clippy --all-targets -- -D warnings
   cargo test --all-targets
   cargo fmt --check
   ```
3. **Write tests** for new functionality, especially scheduler and model code.
4. **Update documentation** if you change public APIs or add features.
5. **Add a changelog entry** if applicable (see `CHANGELOG.md`).

## Code Style

- **Formatting**: `cargo fmt` (stable Rust style).
- **Linting**: Clippy must pass with `-D warnings`.
- **Unwraps**: Avoid `.unwrap()` in production code. Use `?` or `.context()` / `.expect()` with a meaningful message.
- **Dead code**: Prefer item-level `#[allow(dead_code)]` over file-level `#![allow(dead_code)]`.
- **Imports**: Group by std → external → crate. One blank line between groups.
- **Naming**: Descriptive names. Use `_` prefix for intentionally unused variables.
- **Errors**: Use `anyhow::Context` for contextual errors; define domain error types with `thiserror` for public APIs.

## Testing

- **Unit tests**: Use `#[cfg(test)] mod tests { ... }` adjacent to the tested module.
- **Integration tests**: Place in `tests/` directory at the project root.
- **Coverage**: All new `pub fn` items should have at least one unit test.
- **Property-based testing**: Consider `proptest` for scheduler and cache logic.

## Developer Certificate of Origin (DCO)

By contributing to Kyro, you certify that:

- The contribution was created in whole or in part by you and you have the right to submit it under the Apache 2.0 license; or
- The contribution is based upon previous work that, to the best of your knowledge, is covered under an appropriate open source license and you have the right under that license to submit that work with modifications; or
- The contribution was provided directly to you by some other person who certified the above.

To acknowledge the DCO, sign off your commits:

```bash
git commit -s -m "feat: add new feature"
```

This adds a `Signed-off-by: Your Name <your.email@example.com>` trailer to the commit message.

## Questions?

Open a [Discussion](https://github.com/nrelab/kyro/discussions) or join the maintainers on the issue tracker.
