# Changelog

All notable changes are documented here. This project follows Semantic
Versioning after the first stable release.

## [Unreleased]

## [0.1.0] - 2026-07-14

### Added

- Native Apple Silicon inference through MLX and Metal.
- English-to-Chinese Marian Transformer/SSRU greedy decoding with lexical
  shortlist support.
- Rust/Axum HTTP service with bounded admission, dynamic micro-batching,
  health endpoints, Prometheus metrics, and graceful shutdown.
- `/translate`, `/detect`, and Immersive Translate `/imme` request shapes.
- Region-aware language normalization such as `en-US` to `en` and `zh-CN` to
  `zh`.
- Rootless launchd installer and CPU-only multi-architecture Docker path.

[Unreleased]: https://github.com/malusama/marian-mlx/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/malusama/marian-mlx/releases/tag/v0.1.0
