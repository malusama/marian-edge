# Changelog

All notable changes are documented here. This project follows Semantic
Versioning after the first stable release.

## [Unreleased]

No changes yet.

## [0.4.0] - 2026-07-15

### Added

- Added a fused, forward-only FlashAttention-style Metal kernel. It processes
  four query rows per SIMD group, streams 32-key tiles with online softmax, and
  avoids materializing the quadratic attention-score matrix.
- Added `MARIAN_MLX_METAL_ATTENTION=auto|classic|flash` and
  `MARIAN_MLX_METAL_FLASH_THRESHOLD`; `/info` now exposes the selected
  attention implementation.
- Added `--max-output-tokens` to the HTTP benchmark driver for isolated
  encoder/attention measurements.

### Changed

- Enabled the fused attention path by default for supported self-attention and
  single-query cross-attention shapes, while retaining the classic kernels as
  an explicit comparison and compatibility path.
- Vectorized CPU bias, ReLU, SSRU residual, softmax scaling, and attention-value
  accumulation with NEON or runtime-gated AVX2. Numerically sensitive dot,
  softmax, sigmoid, and normalization reductions retain their scalar order.

### Performance

- On Apple M1, fused attention improved the documented concurrent short-text
  FP32 workload by 2.8% and the 200-item workload by 3.5% versus the same
  direct-Metal runtime's classic attention path.
- On an encoder-isolated long-sequence workload, fused attention reduced p50
  by 23.8% at roughly 40 repeated phrases and by 26.5% at 320 repeated phrases.
- Re-measured the first v0.1.0 MLX release and the optimized direct-Metal
  runtime under identical settings, including the remaining short-workload
  performance gap and v0.1.0's nondeterministic repeated corpus outputs.

## [0.3.0] - 2026-07-15

### Added

- Added a checked-in deterministic 200-item corpus, reproducible HTTP benchmark
  driver, CPU hotspot microbenchmarks, and a repeatable Instruments Metal
  System Trace helper.
- Added an explicit `MARIAN_MLX_METAL_PRECISION=mixed-f16` Metal model-storage
  mode. FP32 remains the default; the selected precision is exposed by
  `/info`, and the qualification test reports corpus-level token differences.
- Added Q8 memory reporting for canonical weights, packed weights, embeddings,
  and packed-weight construction time.

### Changed

- Reused Q8 activation, accumulator, decoder, attention, shortlist, and tensor
  buffers; added allocation-free `run_into` paths and direct embedding-row
  dequantization.
- Added exact-order NEON/AVX2 residual addition, exact SIMD tail coverage, and
  measured work thresholds that keep small decoder GEMV on the owner thread.
- Reused Metal buffers across decoder steps, fused FFN bias and ReLU into the
  matrix kernel, and aligned Metal long-text segmentation and output-budget
  behavior with the CPU backend.
- Load FP32 safetensors and Q8 artifacts through read-only memory maps before
  transferring data into owned runtime storage.
- Made local builds refresh the embedded Git revision after the current branch
  advances, so `/info` does not retain a stale cached commit.

### Performance

- On the documented Apple M1 run, explicit mixed-f16 storage improved direct
  Metal throughput by 23.4% for the concurrent single-sentence workload and
  4.2% for the 200-item corpus while reducing peak RSS by 25.2% and 35.2%.
- Q8 allocation changes kept measured throughput effectively flat while
  reducing peak RSS by 21.3% for the warm single-sentence run and 4.7% for the
  200-item corpus.

## [0.2.1] - 2026-07-15

### Fixed

- Preserved saved port and CORS settings when the macOS installer is re-run
  without explicit overrides; an explicitly empty CORS override still clears
  the saved origin.
- Rebuilt the LaunchAgent from the target release template during rollback so
  switching between the legacy MLX layout and the direct Metal layout does not
  leave duplicate backend arguments.

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

[Unreleased]: https://github.com/malusama/marian-mlx/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/malusama/marian-mlx/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/malusama/marian-mlx/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/malusama/marian-mlx/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/malusama/marian-mlx/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/malusama/marian-mlx/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/malusama/marian-mlx/releases/tag/v0.1.0
