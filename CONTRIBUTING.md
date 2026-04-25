# Contributing to Kyro

Thank you for your interest in Kyro! We welcome contributions from the community to make Kyro the fastest and most reliable LLM serving engine.

## How to Contribute

### 1. Reporting Bugs
- Use the GitHub Issue tracker.
- Provide a clear description and a minimal reproducible example.

### 2. Feature Requests
- Open an issue to discuss major features before implementing them.

### 3. Pull Requests
- Ensure your code follows the standard Rust formatting (`cargo fmt`).
- Add tests for any new functionality.
- Keep PRs focused on a single change.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/nrelab/kyro.git
cd kyro

# Build for development
cargo build

# Run benchmarks to ensure no performance regressions
python benchmarks/stress_test.py
```

## Community & Conduct
Please be respectful and professional in all interactions within the Kyro community.
