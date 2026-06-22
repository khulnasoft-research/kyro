# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

To report a security vulnerability, please open a confidential issue or contact the maintainers directly.

**Do not** file a public issue for security vulnerabilities.

We will acknowledge receipt within 48 hours and provide a timeline for resolution.

## Supply Chain Security

- Dependencies are audited via `cargo audit` in CI.
- Dependabot is configured for automated dependency update PRs.
- All commits must be signed off (DCO).
- The Docker image runs as a non-root user and is built from `debian:bookworm-slim`.
