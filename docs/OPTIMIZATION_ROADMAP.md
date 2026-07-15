# Optimization roadmap

This document separates what the runtime does today, what was investigated and
rejected, and what still requires implementation or hardware qualification.
Code completion and measured performance are deliberately different states.

## Qualified release: v0.6.0

The final v0.6.0 release candidate was compared live with freshly built v0.1.0
and v0.5.0 binaries on the documented Apple M1 / 16 GB host. The candidate's
version field still reported 0.5.0; the v0.6.0 release uses the same inference
source. Under visible desktop load, paired three-run medians were:

| FP32 workload | v0.1.0 | v0.5.0 live | v0.6 candidate | v0.6 vs v0.1 |
|---|---:|---:|---:|---:|
| 1,000 repeated short requests | 486.64 item/s | 496.14 item/s | 546.19 item/s | +12.2% |
| 5 x 200 distinct corpus items | 116.68 item/s | 126.37 item/s | 149.14 item/s | +27.8% |

Metal FP32 and CPU FP32 match on 200/200 deterministic corpus items. Release
goldens pass, mixed-F16 remains explicitly 198/200, Flash and classic output
hashes match, and the final Metal trace contains 40/40 completed command
buffers with zero errors. Exact settings, per-run values, hashes, latency,
memory, trace evidence, and quieter historical peaks are in
[`BENCHMARKS.md`](BENCHMARKS.md).

## Current implementation

### Measurement and correctness contract

- A checked-in deterministic 200-item corpus and generator cover distinct
  production-like inputs rather than only repeated cache-friendly text.
- `tools/bench_http.py` records commit, backend, model metadata, concurrency,
  warmup, latency percentiles, throughput, and peak RSS.
- CPU hotspot benchmarks isolate Q8 GEMV/GEMM, shortlist, elementwise,
  attention, tokenizer, and segmentation work.
- Container smoke tests cover native Arm64 and AMD64 startup; build success or
  Rosetta execution is not treated as native SIMD performance evidence.
- Golden, differential, newline/CRLF, long-input, padding, and segmented
  `max_output_tokens` tests protect the output contract.
- Metal FP32, mixed storage, Flash, and classic paths remain separately
  selectable so memory, speed, and token differences are reportable.

### Shared runtime and scheduling

- `marian-core` owns the tokenizer-aware segmentation algorithm. CPU and Metal
  supply exact tokenizer counts and backend-specific segment limits.
- `Translator::translate_many` owns canonical batch ordering and restores
  caller order for bounded multi-item requests such as `/imme`.
- Core coalesces byte-identical logical rows only inside the current dynamic
  batch and passes repetition counts through the backend contract.
- Accelerator backends decide whether repetitions should become extra physical
  rows. Device occupancy widths do not live in the generic scheduler, and
  there is no cross-batch result cache.

### CPU

- Q8 dense weights remain quantized. Engine-owned linear scratch, reusable
  tensor/attention/decoder/shortlist buffers, direct embedding dequantization,
  and `run_into` APIs remove hot-path allocation churn.
- Runtime reporting separates canonical, packed, and embedding bytes and
  exposes packed-weight build time so representation cost is measurable.
- Read-only mmap is used while loading FP32 safetensors and Q8 artifacts before
  data moves into the runtime's owned representation.
- Arm `SDOT`, exact widening AVX2, and scalar fallbacks cover Q8 work without an
  i16-saturating intermediate. Tail and signed-range oracles protect dispatch.
- NEON and runtime-gated AVX2 cover independent elementwise work. Sensitive
  dot, normalization, softmax-sum, and sigmoid/state reductions retain their
  scalar order.
- Shape thresholds keep small fixed-width decoder work on the owner thread when
  Rayon overhead would dominate. `--cpu-threads` changes internal pools, not
  model ownership or weight count.

### Metal

- FP32 dense operations use shape-cached MPS matrix multiplication. The
  explicit `mixed-f16` storage mode can use a custom 32x32 microtile; it does
  not silently replace the FP32 output contract.
- Supported attention shapes use a four-query tiled, online-softmax
  FlashAttention-style kernel that does not materialize the O(N^2) score
  matrix. Classic attention remains the compatibility and A/B fallback.
- The fused encoder and cross-cache build reduce command-buffer transitions;
  decoder logits, argmax, state advancement, EOS/limit handling, history, and
  bounded multi-token submissions stay on GPU.
- Encoder QKV, decoder cross K/V, and SSRU W/Wf projections are packed so each
  pair or triple uses one matrix multiplication. Output projection bias,
  residual, and layer normalization share one kernel.
- `MetalWorkspace` centralizes request and transient buffer arenas.
  `MetalRuntime` keeps permanent model-weight MPS views separate from
  request-scoped arena views and clears the latter at each request boundary.
- `MetalTuning` centralizes attention, GEMM, decode, and duplicate-row policy
  after device selection. The current M1 defaults are measured in the v0.6.0
  release qualification; M2, M3, M4, and generic profiles are conservative
  starting points only.
- Decode submission length is selected from active rows, remaining steps,
  observed completion, and the active profile rather than a fixed global
  constant.

These workspace, cache-lifetime, tuning, scheduler-boundary, fusion, and packed
projection changes passed the fresh benchmark and correctness matrix above.

## Evaluated but not shipped

| Candidate | Decision | Reason |
|---|---|---|
| Skip artifact hashing through a local metadata cache | Not shipped | File metadata alone is not a trusted identity; skipping the full checksum would weaken startup validation. |
| Serialize `rten-gemm` packed weights | Not shipped | `rten-gemm 0.21` has no safe public constructor for a versioned serialized `PackedBMatrix`; an unsupported private format would be brittle. |
| Reorder dot, softmax, normalization, or sigmoid reductions | Not shipped | Addition order and transcendental approximations can change greedy token selection; this needs an explicit tolerance and corpus contract. |
| Route full-range Q8 through a saturating i16 intermediate | Rejected | It cannot preserve the full signed-weight integer oracle. |
| Treat M1 tile/occupancy results as M2-M4 optima | Rejected | Family-specific GPU behavior requires measurements on the target device. |
| Adopt MLA, DSA, MTP, FP8, or similar techniques as simple runtime flags | Not a drop-in runtime optimization | They change training, weights, graph semantics, or numerical contracts and require a compatible model plus separate quality qualification. |

## DeepSeek systems review

The transferable lesson from DeepSeek's recent work is hardware/model/runtime
co-design, not copying model-specific acronyms into a fixed Marian checkpoint.
The v0.6.0 work applies the compatible systems ideas: online-softmax attention,
packed projection layouts, fused graph boundaries, explicit device profiles,
bounded multi-step decode submissions, and profiler-backed tuning. This follows
the measurement-first direction in DeepSeek's
[hardware co-design report](https://arxiv.org/abs/2505.09343) and the dense
attention principles exposed by the official
[FlashMLA kernels](https://github.com/deepseek-ai/FlashMLA), while implementing
standard MHA for Apple Metal rather than pretending this Marian model is MLA.

The remaining DeepSeek techniques are different model contracts:

| Technique | Why it is not a v0.6.0 runtime patch |
|---|---|
| MLA / FP8 KV cache | Changes Q/K/V weight shapes, cache representation, precision, and target hardware assumptions. |
| DeepSeek Sparse Attention | The [V3.2 design](https://arxiv.org/abs/2512.02556) depends on a trained token indexer and targets long-context scaling; this Marian checkpoint is dense and capped at 4096 positions. |
| Multi-Token Prediction | Requires compatible prediction heads and training; submitting several existing autoregressive steps together is not MTP or speculative decoding. |
| Engram | [Conditional memory](https://github.com/deepseek-ai/Engram) adds learned/static lookup modules during model design and pretraining. |
| Expert parallelism / DeepEP | This is a single dense model on one integrated GPU, with no experts or inter-device all-to-all traffic. |

Adopt one of these only with a new checkpoint, manifest/graph version, quality
corpus, and separate release qualification. They are not hidden unfinished
optimizations in the current runtime.

## Future work

Everything in the former P0 requalification gate is complete: live v0.1/v0.5
A/B, FP32/mixed correctness, Flash/classic A/B, duplicate and distinct traffic,
and a labeled Metal System Trace. Remaining work requires different physical
hardware, a different model contract, or new profiler evidence; it is not an
unfinished v0.6.0 promise.

### P1: qualify each Apple GPU family

- Run checked-in sweeps on physical M2, M3, and M4 machines for Flash query
  tile, duplicate physical width, decode row budget/step cap, and the custom
  FP32 GEMM row cutoff.
- Promote a family profile from conservative to qualified only when the
  benchmark report includes device name, OS, commit, model hashes, quality
  delta, latency distribution, throughput, and peak RSS.
- Prefer profile-data changes over scattering family checks through graph code.

### P2: native x86-64 performance

- Benchmark exact AVX2 on native AMD64 hardware; tune decoder GEMV and encoder
  GEMM thresholds there rather than under Rosetta.
- Add VNNI or AVX-512 only behind runtime dispatch and the full signed-range
  integer oracle.
- Retain scalar fallback coverage for older x86-64 machines.

### P3: profiler-led fusion

- Consider additional bias, activation, residual, or normalization fusion only
  when a current trace shows material intermediate traffic or submission cost.
- Consider larger or persistent decode submissions only when active-row and EOS
  behavior prove that wasted work is bounded.
- Any new precision or reduction order must be an explicit mode with golden,
  differential, and corpus-level quality gates.
- Revisit a checksum-bound packed-weight cache only if startup time becomes a
  measured deployment bottleneck and the dependency exposes a supported format.

## Reproduction commands

Useful CPU and HTTP probes:

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
```

Useful Metal A/B and trace controls:

```sh
MARIAN_EDGE_METAL_PROFILE=m1 \
MARIAN_EDGE_METAL_ATTENTION=classic \
  target/release/marian-edge-server --backend metal --model-dir models/enzh

MARIAN_EDGE_METAL_PROFILE=m1 \
MARIAN_EDGE_METAL_ATTENTION=auto \
MARIAN_EDGE_METAL_FLASH_THRESHOLD=1 \
  target/release/marian-edge-server --backend metal --model-dir models/enzh

tools/profile_metal.sh \
  http://127.0.0.1:3000/translate /tmp/marian-metal.trace
```

## Merge gates

An optimization is complete only when every relevant gate passes:

- Rust 1.86 formatting, Clippy with warnings denied, workspace tests, and both
  CPU feature test sets;
- exact Q8 integer oracles across the full signed weight range and SIMD tails;
- all five release golden translations;
- before/after comparison on the deterministic 200-item corpus;
- long-input, CRLF/newline preservation, and segmented
  `max_output_tokens`-budget tests;
- CPU FP32 versus Metal FP32 differential tests for shared graph semantics;
- native Arm64 and AMD64 container startup from empty and reused model volumes,
  plus a dynamic-link check proving no C++ or native SentencePiece dependency;
- a benchmark artifact containing commit, model hashes, hardware, OS, thread
  count, warmup, latency distribution, throughput, and peak RSS, including
  regressions rather than only wins.

Do not call the runtime globally optimal from source inspection. After each
material change, re-profile the complete request path and choose the next
bottleneck from evidence.
