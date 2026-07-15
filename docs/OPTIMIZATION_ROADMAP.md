# Optimization roadmap

This document is the handoff for performance work after the pure-Rust runtime
migration. The current implementation is correct and already uses important
architecture-specific kernels, but it is not claimed to be globally optimal.
Apple M1 and later Arm processors are the first optimization target; portable
x86-64 and direct Metal remain required release paths.

## Current baseline (2026-07-15)

- macOS production inference uses a Rust host and embedded MSL on direct Metal.
- Supported Metal attention shapes use a fused four-query online-softmax
  kernel by default and do not allocate an O(N^2) score matrix.
- The FP32 CPU graph uses `matrixmultiply`.
- The Q8 CPU graph keeps dense weights quantized and uses `rten-gemm`.
- On the measured M1 Pro, `rten-gemm` selects
  `aarch64-u8i8i32-dotprod`: NEON and Arm DotProd are present; I8MM is not.
- Sparse decoder shortlist scoring uses a runtime-gated Arm `SDOT` kernel with
  a scalar exact fallback.
- Full-range x86 Q8 uses the repository's widening AVX2 kernel instead of a
  saturating `vpmaddubsw` path; machines without AVX2 use an exact scalar
  fallback.
- CPU elementwise bias, ReLU, residual, softmax scaling, and attention-value
  loops use NEON or runtime-gated AVX2 without reordering sensitive reductions.
- `--cpu-threads` configures the Rayon and FP32 matrix-multiply pools, while
  model ownership and weights remain single-copy.

The M1 `SDOT` shortlist change preserved every token in the 200-item
differential corpus and all five release golden translations. On the same M1
Pro with one CPU thread:

| Workload | Before | After | Change |
|---|---:|---:|---:|
| warm single sentence, p50 | 10.90 ms | 5.17 ms | -52.6% |
| one 200-item `/imme` request | 1349.6 ms | 897.8 ms | -33.5% |
| 200-item effective throughput | 148.2 sentence/s | 222.8 sentence/s | +50.3% |

After that change, measured 1/2/4-thread average latency for the warm weather
sentence was 5.12/7.48/8.18 ms. The corresponding 200-item request averages
were 937.7/1354.5/1243.9 ms, with sampled peak RSS around
72.9/72.3/73.8 MiB. This fixed 384-dimensional graph is too small to amortize
the current Rayon overhead at higher thread counts, so one thread remains the
default. These are engineering measurements, not published product claims.

## Implementation status (commit `79e81466f418`)

Every item below has been implemented or evaluated. Hardware-specific claims
remain deliberately separate from code completion.

| Roadmap item | Status | Result / boundary |
|---|---|---|
| P0 deterministic measurements | complete | Checked-in 200-item corpus and generator; metadata-complete HTTP driver; JSON microbenchmarks; Arm64/AMD64 container smoke; segmented output-budget tests. |
| P1 Q8 allocation/data flow | complete | Engine-owned linear scratch, reusable tensor/attention/shortlist buffers, direct embedding dequantization, `run_into` APIs, and measured Rayon thresholds. |
| P1 Q8 representation cost | complete | Runtime reports canonical, packed, and embedding bytes plus packed-build time; real artifact result is recorded in `BENCHMARKS.md`. |
| P1 direct Metal | complete on M1 | Existing tiled matmul retained; FFN bias+ReLU fused; reusable buffer arena; long-text semantics unified; explicit mixed-f16 storage mode; current Metal System Trace captured. |
| P1 fused attention | complete on M1 | Four-query/32-key tiled Metal kernel, online softmax, no O(N^2) score buffer, padding coverage, classic fallback, environment-controlled A/B path, real-model corpus qualification. |
| M2/M3/M4 Metal tile tuning | hardware validation pending | The implementation and trace tooling are ready, but an M1 cannot establish family-specific optimum tile sizes. Do not copy the M1 result into M2-M4 claims. |
| P2 Arm/x86 safe FP32 SIMD | complete | NEON and runtime-gated AVX2 bias, ReLU, SSRU residual, softmax scaling, and attention-value accumulation with every-tail tests. Dot, sigmoid, softmax-sum, and normalization reductions remain scalar by correctness contract. |
| P2 x86-64 | code complete; native performance pending | Exact widening AVX2, scalar fallback, tail oracle, and output-work threshold are covered. CI proves native startup; native AMD64 performance and optional VNNI/AVX-512 still require corresponding hardware. |
| P3 read-only mapping | complete | FP32 safetensors and Q8 artifacts load through read-only mmap, then move into owned CPU or Metal storage. |
| P3 checksum metadata cache | evaluated, not shipped | Skipping a full hash without a trusted external identity would weaken startup validation, violating the merge constraint. Downloads and activation remain atomic. |
| P3 serialized packed cache | evaluated, not shipped | `rten-gemm 0.21` exposes no safe public constructor for a serialized `PackedBMatrix`; startup build time is measured instead of introducing an unsupported binary format. |

All implementation items that can be qualified on the current M1 are complete.
The remaining entries are hardware-specific validation or new arithmetic
contracts: M2-M4 tile selection, native AMD64 measurements, optional VNNI or
AVX-512 kernels, and any reduction-order change. Those cannot be honestly
closed from an M1 result. The current M1 results, quality deltas, and profiler evidence are in
[`BENCHMARKS.md`](BENCHMARKS.md).

## P0: keep measurements and output contracts reproducible

Before changing arithmetic or allocation strategy:

1. Add a checked-in deterministic corpus generator and a benchmark driver that
   records commit, model checksums, backend, kernel name, host, thread count,
   warmup, p50/p95/p99, throughput, and peak RSS.
2. Keep separate microbenchmarks for Q8 GEMV/GEMM, shortlist scoring,
   attention, layer normalization, SSRU, tokenizer, and long-text planning.
3. Run native Arm64 and AMD64 container smoke tests. Do not treat cross-compile
   success or Rosetta fallback execution as SIMD runtime proof.
4. Preserve the original `TranslationInput::max_output_tokens` budget across
   all automatically created text segments.

Useful probes:

```sh
cargo +1.86.0 test --locked -p marian-cpu \
  q8_arm::tests::rten_selects_the_best_available_arm_kernel -- --nocapture

MARIAN_Q8_MODEL=/path/to/model.q8.bin \
MARIAN_CPU_MODEL_DIR=models/enzh \
cargo +1.86.0 test --locked --release -p marian-cpu \
  --test q8_golden -- --ignored --nocapture

python3 tools/bench_http.py --url http://127.0.0.1:3000/translate \
  --requests 500 --concurrency 1 --warmup 32 \
  --model-dir models/enzh --pid "$(lsof -tiTCP:3000 -sTCP:LISTEN)" \
  --commit "$(git rev-parse HEAD)" --output benchmark.json

cargo +1.86.0 bench --locked -p marian-cpu \
  --features benchmarks --bench hotspots

tools/profile_metal.sh \
  http://127.0.0.1:3000/translate /tmp/marian-metal.trace
```

## P1: M1/M-series Q8 CPU allocation and data flow

These changes should preserve integer results and token output exactly.

1. Introduce an engine-owned scratch arena. `Q8Linear::run` currently allocates
   quantized activations, accumulators, and FP32 output for every call; one
   decoder token performs roughly two dozen linear operations. Add `run_into`
   APIs and reuse capacity by shape.
2. Dequantize embedding rows directly into their destination slice. Avoid the
   temporary `Vec<f32>` currently returned by `Q8Embedding::row`.
3. Reuse decoder, residual, attention, and shortlist candidate buffers across
   generation steps. Establish explicit aliasing rules before adding in-place
   operations.
4. Add shape thresholds for Rayon. Small GEMV and 384-dimensional single-item
   paths should stay on the owner thread; parallelism should be enabled only
   when measured row/output work amortizes scheduling.
5. Measure the memory trade-off of retaining both canonical Q8 weights and
   `rten-gemm` packed weights. Canonical rows are currently useful for GEMV and
   shortlist access, but the duplicate representation should be quantified.

Primary files:

- `crates/marian-cpu/src/q8_gemm.rs`
- `crates/marian-cpu/src/q8_engine.rs`
- `crates/marian-cpu/src/q8_arm.rs`
- `crates/marian-cpu/src/tensor.rs`

## P1: direct Metal on Apple Silicon

Metal remains the preferred macOS production path. Profile GPU time and command
submission separately before changing kernels.

1. Tile dense matrix multiplication in threadgroup memory and tune tile sizes
   on M1, M2, M3, and M4 rather than assuming one family represents all Apple
   GPUs.
2. Fuse bias, activation, residual, and layer-normalization stages where this
   removes intermediate buffers and command encoders.
3. Add a reusable Metal buffer arena and encode a whole decoder step with fewer
   allocations and command-buffer transitions.
4. Evaluate FP16 or mixed-precision storage only as an explicit new precision
   mode. It must not silently replace the FP32 output contract, and it needs
   corpus-level quality and token-difference reporting.
5. Capture Metal System Trace evidence for GPU occupancy, memory bandwidth,
   threadgroup pressure, and CPU submission gaps.
6. Fuse attention score, masking, online softmax, and value accumulation so
   supported shapes stream K/V tiles without a quadratic score allocation.

Primary files:

- `crates/marian-mlx/metal/kernels.metal`
- `crates/marian-mlx/src/metal_runtime.rs`
- `crates/marian-mlx/src/engine.rs`

## P2: scalar reduction boundary

The safe elementwise work is implemented. Attention score dots, softmax
maximum/sum, layer-normalization statistics, and SSRU sigmoid/state updates
remain scalar because changing their addition or transcendental order changes
the numerical contract. This is a deliberate correctness boundary rather than
an accidentally unfinished elementwise optimization.

Implemented operations:

- NEON and runtime-gated AVX2 bias addition, ReLU, SSRU ReLU/residual, softmax
  scaling, and attention score-times-value accumulation.
- Exact scalar key order is retained while independent output dimensions are
  vectorized.
- Tail lengths 0 through 65 are checked against scalar oracles.

Possible future work requires an explicit new tolerance and corpus gate:

- compensated or reordered layer-normalization reductions;
- vectorized attention-score dot products and softmax reductions;
- alternative sigmoid approximations;
- fused Q8 output arithmetic if it can preserve quantization rounding.

## P2: x86-64 Q8

1. Benchmark the exact AVX2 path on native AMD64 hardware; Rosetta does not
   expose AVX2 and therefore validates only dispatch/fallback behavior.
2. Tune output-channel tiling and parallel thresholds for single-row decoder
   GEMV as well as encoder GEMM.
3. Consider VNNI/AVX-512 runtime paths only with a non-saturating, full-range
   integer oracle. Never route full Marian Q8 weights through an i16-saturating
   intermediate.
4. Keep scalar fallback coverage for older x86-64 machines.

## P3: loading and memory

- Evaluate read-only memory mapping for FP32 safetensors and Q8 artifacts.
- Cache verified manifest/checksum metadata without weakening startup
  validation.
- Measure packed-weight creation time and consider a versioned, checksum-bound
  cache only if startup becomes material.
- Keep model downloads outside the image and preserve atomic model activation.

## Merge gates for every optimization

An optimization is complete only when the relevant gates pass:

- Rust 1.86 formatting, Clippy with warnings denied, workspace tests, and both
  CPU feature tests.
- Exact Q8 integer oracle tests, including full signed weight range and every
  SIMD tail.
- Five release golden translations.
- Before/after token comparison on the deterministic 200-item corpus.
- The 80-sentence long-input case, CRLF/newline preservation, and a segmented
  `max_output_tokens` budget test.
- FP32 CPU versus Metal differential tests for changes touching shared graph
  semantics.
- Native Arm64 and AMD64 container startup from an empty model volume, restart
  with the existing volume, and a dynamic-link check showing no C++ or native
  SentencePiece dependency.
- A benchmark report containing the exact commit, model hashes, hardware,
  thread count, warmup, latency distribution, throughput, and peak RSS. Report
  regressions as well as improvements.

Do not label the implementation “optimal” from instruction inspection alone.
Use the profiler to identify the next bottleneck after every material change.
