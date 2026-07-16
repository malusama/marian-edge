# Optimization roadmap

This file lists unfinished work. Completed measurements belong in
[BENCHMARKS.md](BENCHMARKS.md), and implementation details belong in
[ARCHITECTURE.md](ARCHITECTURE.md).

## Baseline

The current release has:

- pure-Rust FP32 and Q8 CPU backends;
- a direct Metal FP32 backend with optional mixed-F16 weight storage;
- fused online-softmax attention on supported Metal shapes;
- dynamic batching, duplicate coalescing, and tokenizer-aware long-text
  splitting;
- deterministic CPU FP32 versus Metal FP32 output on the checked-in 200-item
  corpus.

The published decoder is greedy: it keeps one hypothesis per input
(`beam=1`).

## How to evaluate a change

Every performance result must record the commit, model hashes, hardware, OS,
precision, request corpus, warmup, concurrency, latency distribution,
throughput, and peak RSS. Compare output on the release goldens and the
200-item corpus before comparing speed.

Hardware-specific results stay hardware-specific. In particular, M1 tuning is
not a claim about M2-M4, and Rosetta is not an AMD64 SIMD benchmark.

## P1: beam-search quality experiment

`beam>1` may recover a better sentence after an early token choice, but it is
not automatically better for this model or this product. It also expands
decoder work and state while leaving encoder work mostly unchanged.

The Mozilla browser inference configuration for this `base-memory` model also
uses `beam=1`, so a wider beam starts as an experiment rather than a presumed
upgrade. An exploratory Marian reference run measured beam 2 at +33.5% elapsed
time for +0.7 BLEU/+0.4 chrF2++, while beam 4 took +88.2% for only another
+0.2/+0.2; spot checks still found mixed changes. See
[the beam-width notes](BENCHMARKS.md#exploratory-beam-width-reference-2026-07-16).
Compare beam sizes `1`, `2`, and `4` first:

1. Use the same fixed English-to-Chinese evaluation set for every beam size.
   Include short web text, long sentences, ambiguity, placeholders, numbers,
   names, and punctuation.
2. Record automatic metrics and a blinded human preference sample. Exact-match
   against beam 1 is diagnostic only; it is not a quality score.
3. Measure single-request latency, batched throughput, peak memory, output
   length, and early-EOS rate on CPU and Metal.
4. Start any optional quality mode at `beam=2`. Use `beam=4` as the upper
   comparison; test 8 only if quality is still improving at 4.

Implementation requires more than replacing argmax with top-k:

- keep K token histories, cumulative log probabilities, and SSRU states;
- expand K hypotheses against the lexical shortlist and retain the global top
  K after each step;
- reorder decoder state when beam rows are reordered;
- define EOS ranking, length normalization, and tie behavior;
- keep CPU FP32, Q8, and Metal behavior covered by golden and differential
  tests;
- reject unsupported `beam` request values instead of silently ignoring them.

Keep `beam=1` as the default for the local webpage-translation target. Offer a
beam-2 quality mode only after the Rust implementation matches the reference,
passes a blinded webpage sample, and has acceptable CPU and Metal latency.

## P2: Apple GPU profiles

Run the checked-in sweeps on physical M2, M3, and M4 machines. Measure Flash
query tile, duplicate physical width, decode row budget, decode steps per
submission, selection threads, and the custom GEMM cutoff.

Promote a device profile only when its benchmark report includes output checks
and a Metal trace. Keep profile values in `MetalTuning`; do not scatter device
checks through the graph code.

## P3: native x86-64 CPU

- Benchmark AVX2 on native AMD64 hardware and tune decoder GEMV and encoder
  GEMM thresholds there.
- Consider VNNI or AVX-512 behind runtime dispatch and the existing signed-range
  integer oracle.
- Keep scalar fallbacks for older x86-64 hosts.

## P4: profiler-led Metal work

- Add kernel fusion only when a current trace shows material intermediate
  traffic or submission overhead.
- Test larger or persistent decode submissions against active-row and EOS
  behavior; avoid doing work for completed rows.
- Treat any new precision or reduction order as a separate mode with golden,
  differential, and corpus-level output checks.

## P5: startup time

Revisit a packed-weight cache only if startup becomes a measured deployment
bottleneck and the dependency exposes a supported, versioned representation.
Do not skip model checksums based only on file metadata.

## Deferred ideas

| Idea | Why it is deferred |
|---|---|
| Serialize `rten-gemm` packed weights | `rten-gemm 0.21` has no supported constructor for a versioned serialized `PackedBMatrix`. |
| Route full-range Q8 through a saturating i16 intermediate | It does not preserve the signed-weight integer result. |
| Reorder sensitive reductions for speed | It can change greedy token selection and needs a separately qualified numerical policy. |
| MLA, FP8 KV cache, sparse attention, or multi-token prediction | These require different weights, graph structure, or training; they are not flags for the current checkpoint. |

## Reproduction commands

CPU checks:

```sh
cargo +1.86.0 test --locked -p marian-cpu \
  q8_arm::tests::rten_selects_the_best_available_arm_kernel -- --nocapture

MARIAN_Q8_MODEL=/path/to/model.q8.bin \
MARIAN_CPU_MODEL_DIR=models/enzh \
cargo +1.86.0 test --locked --release -p marian-cpu \
  --test q8_golden -- --ignored --nocapture

cargo +1.86.0 bench --locked -p marian-cpu \
  --features benchmarks --bench hotspots
```

HTTP benchmark:

```sh
SERVICE_ORIGIN=${SERVICE_ORIGIN:-http://127.0.0.1:3000}
PORT=${SERVICE_ORIGIN##*:}
python3 tools/bench_http.py --url "$SERVICE_ORIGIN/translate" \
  --requests 500 --concurrency 1 --warmup 32 \
  --model-dir models/enzh --pid "$(lsof -tiTCP:"$PORT" -sTCP:LISTEN)" \
  --commit "$(git rev-parse HEAD)" --output benchmark.json
```

Metal A/B and trace:

```sh
MARIAN_EDGE_METAL_ATTENTION=classic \
  target/release/marian-edge-server --backend metal --model-dir models/enzh

MARIAN_EDGE_METAL_ATTENTION=auto \
  target/release/marian-edge-server --backend metal --model-dir models/enzh

SERVICE_ORIGIN=${SERVICE_ORIGIN:-http://127.0.0.1:3000}
tools/profile_metal.sh \
  "$SERVICE_ORIGIN/translate" /tmp/marian-metal.trace
```

## Merge checklist

- formatting, Clippy, workspace tests, and CPU feature tests pass;
- release goldens and CPU/Metal differential tests pass;
- long input, newline preservation, and shared `max_output_tokens` behavior pass;
- native AMD64 and ARM64 container smoke tests pass when the change affects
  portable deployment;
- the benchmark artifact includes regressions as well as wins.

After a material change, profile the full request path again before choosing
the next target.
