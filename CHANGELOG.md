# Changelog

All notable changes are documented here. This project follows Semantic
Versioning after the first stable release.

## [Unreleased]

### Changed

- Shortened the READMEs and optimization roadmap, moving implementation and
  benchmark detail to their dedicated documents.
- Reworked the Immersive Translate instructions around one `SERVICE_ORIGIN` so
  health checks and `/imme` cannot silently switch between ports 3000 and 3100.
- Added a beam 1/2/4 reference comparison; beam 1 remains the latency default
  and beam 2 is the candidate for a future quality mode.

### Fixed

- Corrected the documented legacy Metal environment prefix to
  `MARIAN_MLX_METAL_*`.
- Clarified that `beam=1` is a decoder setting, not a model property.

## [0.7.0] - 2026-07-16

### Changed

- Renamed the product and repository to Marian Edge. The executable,
  archive, service, image, controller, and installer now use `marian-edge`;
  the direct-Metal Rust package and source directory use `marian-metal` /
  `crates/marian-metal`.
- Made `marian-edge.transformer-ssru.v1` the canonical manifest namespace for
  new CPU/Metal model producers while loaders, installer, and controller retain
  the historical manifest value. Safetensors metadata stays byte-stable so the
  published FP32 weight checksum does not change.
- Centralized graph validation and sinusoidal positions in `marian-model`, and
  centralized CPU-only execution limits privately in `marian-cpu` instead of
  leaking backend policy into the model schema.
- Made `MARIAN_EDGE_*` the canonical runtime and Metal tuning namespace,
  retained `MARIAN_MLX_*` runtime aliases for migration, and made conflicting
  values fail closed.
- Added a zero-dependency documentation contract check for release versions,
  relative links, public routes, Immersive Translate fields and limits, and
  native/Docker port mappings; CI and release gates now run it.
- Made the Docker Compose host port and image configurable through
  `MARIAN_EDGE_HOST_PORT` and `MARIAN_EDGE_IMAGE` while retaining container port
  3000 and loopback-only host publication by default.

### Fixed

- Switched release panic handling to unwind so the existing HTTP boundary and
  backend-owner guard can turn a panic into terminal not-ready state; added a
  scheduler regression for failed readiness and future admission.
- Added model-backed FP32 and `/imme` adapter regressions for Immersive
  Translate's numbered and paired-tag placeholders.
- Aligned native, Docker, health-check, API, and Immersive Translate examples
  so a custom host port such as 3100 is used consistently end to end.
- Corrected readiness/queue error semantics, complete JSON body and `/imme`
  limits, endpoint field mappings, installed-service lifecycle commands,
  release sequencing, and performance-evidence boundaries across the docs.

## [0.6.0] - 2026-07-16

### Added

- Added public `MetalConfig`, precision/attention/profile enums, and
  `MetalBackend::load_with_config`; process environment parsing now stops at
  the configuration boundary.
- Added explicit M1-M4/generic tuning profiles for Flash query tiles, duplicate
  occupancy, decode row budget/step cap, selection threads, and custom GEMM
  routing. `/info` reports every resolved value.
- Added request/transient `MetalWorkspace` arenas, permanent/request MPS view
  cache classes, bounded shape caches, startup validation for SIMD width,
  thread limits, and threadgroup memory, and deterministic tests for profile
  defaults and overrides.
- Added labeled Metal command buffers plus an Instruments trace tool that uses
  an attach-ready handshake and exports submissions, completions, errors, GPU
  intervals, device metadata, benchmark data, and a compact summary.
- Added a checked-in Apple M1 release-qualification artifact with live v0.1,
  v0.5, Flash/classic, correctness, and trace evidence.

### Changed

- Moved tokenizer-aware long-text segmentation into `marian-core`, so CPU and
  Metal share one policy without a production dependency between backends.
- Kept scheduler batching backend-neutral: core coalesces logical duplicates
  and passes repetition counts, while an accelerator decides whether to
  materialize extra physical rows for device occupancy.
- Routed bounded multi-item submissions through the scheduler's canonical
  batch ordering instead of duplicating bucketing policy in the HTTP adapter.
- Packed encoder QKV, decoder cross K/V, and SSRU W/Wf projections, reducing
  each group to one matrix multiplication while allowing Flash attention to
  consume packed strides and offsets directly.
- Fused output projection bias/residual/layer normalization and fused decoder
  logits, argmax, token advance, EOS/limit tracking, and history recording.
- Made decode submission length respond to active rows, remaining budget, and
  newly observed completion. The qualified M1 profile uses row budget 54, up
  to six steps, and 256 selection threads.
- Split the Metal engine into request/encoder orchestration, artifact and
  packed-weight loading, decode policy, and checked GPU primitives with a
  one-way dependency direction.
- Made release assets and versioned CPU images immutable, generated release
  notes from this changelog, prevented manual container builds from moving
  stable tags, and added Metal Clippy plus profiler-parser gates to CI/release
  workflows.

### Fixed

- Fixed Flash query tiles 1 and 2 dispatching overlapping query rows.
- Fixed long-text output-budget accounting so generated EOS/control tokens do
  not grant each segment a fresh caller budget.
- Fixed request MPS views retaining old arena buffers and per-decode history
  allocations growing to the model's maximum output length.
- Replaced post-MPS compute-encoder `expect` failures with propagated errors;
  Metal command creation or execution failures now mark the backend not-ready,
  while caller validation errors remain non-fatal.
- Synchronized Docker and English/Chinese installation metadata with v0.6.0.

### Performance

- On the live Apple M1 comparison, the final v0.6.0 release candidate reached
  three-run medians of 546.19 item/s for 1,000 repeated short requests and
  149.14 item/s for five 200-item corpus requests: 12.2% and 27.8% above a
  freshly rebuilt v0.1.0 MLX binary. The measured binary still reported 0.5.0;
  the release uses the same inference source.
- Against the same final binary's classic attention path, Flash q4 improved
  short and corpus throughput by 12.3% and 4.9% with identical output hashes.
- Metal FP32 matched CPU FP32 on 200/200 deterministic items. A 300-request
  trace completed 40/40 command buffers with zero GPU errors.

## [0.5.0] - 2026-07-15

### Added

- Added shape-cached Metal Performance Shaders FP32 matrix multiplication and
  a 32x32 custom mixed-F16 microtile kernel.
- Added `MARIAN_MLX_METAL_DUPLICATE_BATCH_WIDTH`; the qualified M1 default
  retains nine identical rows to balance exact batch coalescing with small
  MPS-matrix occupancy.

### Changed

- Submit the complete Flash encoder and cross-cache build together, reuse
  encoder, decoder, cross-cache, and CPU-upload Metal buffers, and cache safe
  MPS matrix views for persistent weights and arenas.
- Decode up to three autoregressive tokens per command-buffer submission while
  keeping token selection, EOS tracking, and decoder state on the GPU.
- Coalesce exact duplicate inputs only within the current dynamic batch; no
  result is cached across batches.
- Group Immersive Translate items by the scheduler's source-length buckets,
  enqueue the complete bounded request together, and restore original response
  order after translation.

### Performance

- On the qualified Apple M1 / 16 GB host, the same 1,000-request FP32 workload
  reached a three-run median of 599.32 item/s, 11.7% above v0.1.0 MLX FP32 and
  83.8% above v0.4.0 direct Metal.
- The same 10 x 200-item corpus reached 165.29 item/s, 35.4% above v0.1.0 and
  105.5% above v0.4.0. CPU FP32 and Metal FP32 remained exact on 200/200
  deterministic corpus items.

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

[Unreleased]: https://github.com/malusama/marian-edge/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/malusama/marian-edge/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/malusama/marian-edge/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/malusama/marian-edge/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/malusama/marian-edge/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/malusama/marian-edge/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/malusama/marian-edge/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/malusama/marian-edge/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/malusama/marian-edge/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/malusama/marian-edge/releases/tag/v0.1.0
