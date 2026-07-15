# Changelog

All notable changes are documented here. This project follows Semantic
Versioning after the first stable release.

## [Unreleased]

No changes yet.

## [0.2.0] - 2026-07-15

### Changed

- Replaced the native MLX/CXX inference bridge with a Rust host using
  `objc2-metal` and runtime-compiled, embedded MSL compute kernels.
- Renamed the native Cargo feature and CLI backend to `metal`; `mlx` remains a
  compatibility alias for existing automation.
- Simplified native macOS releases to one executable with no `libmlx.dylib` or
  external `.metallib`.
- Replaced the native SentencePiece dependency on macOS with the independent
  `marian-tokenizer` crate backed by pure-Rust `sentencepiece-rust` inference.
- Added a portable `cpu` backend implementing the complete FP32 and Q8 Marian
  Transformer/SSRU graphs, lexical shortlist, and greedy decoder in Rust.
- Made the pure-Rust CPU backend the Linux automatic and container runtime. It
  selects Q8 or FP32 from the validated model manifest; `cpu-q8`, `cpu-fp32`,
  and `rust` remain CLI aliases.
- Applied the 1/2/4 CPU thread setting to both FP32 matrix multiplication and
  Q8 rten/exact-AVX2 row parallelism without creating extra model replicas.
- Added strict Marian binary v1 Q8 tensor-set, shape, alpha, and constant
  validation. Dense Q8 weights stay quantized during inference.
- Added pure-Rust, tokenizer-aware long-text segmentation with bounded chunks,
  stable output ordering, and whitespace/newline preservation.
- Qualified Q8 with five exact release golden outputs and a 200-item
  differential corpus: 164/200 outputs exactly matched the retired CPU
  reference. Tested 80-sentence and newline cases matched its long-text
  baseline; general bit-for-bit equivalence is not claimed.
- Added runtime-gated Arm SDOT and exact full-range AVX2 kernels, native AMD64
  and ARM64 container smoke tests, and a measured optimization roadmap.
- Documented the release boundary between source, runtime artifacts, and model
  downloads, with a repeatable release checklist.

### Removed

- Removed the Bergamot/C++ CPU worker, its build toolchain, pipe protocol, and
  container runtime dependencies.

## [0.1.1] - 2026-07-14

### Fixed

- Made converted model artifacts byte-for-byte reproducible for native install
  verification.

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

[Unreleased]: https://github.com/malusama/marian-mlx/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/malusama/marian-mlx/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/malusama/marian-mlx/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/malusama/marian-mlx/releases/tag/v0.1.0
