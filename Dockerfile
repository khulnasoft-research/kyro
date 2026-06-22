# syntax=docker/dockerfile:1
# --- Stage 1: Builder ---
FROM rust:nightly-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    cmake \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/kyro

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Create dummy source to pre-build dependencies (caching)
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release 2>/dev/null || true

# Copy actual source
COPY . .

# Build the real binary with optimizations
RUN cargo build --release

# --- Stage 2: Runtime ---
FROM debian:bookworm-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -s /bin/false kyro

# Copy binary from builder
COPY --from=builder /usr/src/kyro/target/release/kyro /usr/local/bin/kyro

# Use non-root user
USER kyro
WORKDIR /app

# Expose API port
EXPOSE 3000

# Metadata
LABEL org.opencontainers.image.title="Kyro LLM Engine" \
      org.opencontainers.image.description="A high-performance ML inference engine" \
      org.opencontainers.image.source="https://github.com/nrelab/kyro" \
      org.opencontainers.image.licenses="Apache-2.0"

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -sf http://localhost:3000/health || exit 1

# Run the engine
ENTRYPOINT ["kyro"]
