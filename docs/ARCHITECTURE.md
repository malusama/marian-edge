# Architecture and maintenance guide

This document explains where a change belongs and which invariants must remain
true.

## Request path

```text
HTTP clients
    |
    v
Axum validation and compatibility adapters
    |
    v
bounded flume queue -- full --> HTTP 503 + Retry-After
    |
    v
direction/shape-aware micro-batcher
    |
    v
one backend-owner OS thread
    |
    +--> Rust host / embedded MSL --> Metal queue (native macOS)
    |
    +--> pure Rust Q8/FP32 graph / CPU kernels (portable and Linux image)
```

`marian-server` owns HTTP contracts, input limits, CORS, health endpoints, and
shutdown. `marian-core` owns backend-neutral scheduling and metrics.
`marian-tokenizer` owns the narrow pure-Rust SentencePiece inference boundary.
`marian-mlx` owns model validation, the Rust Metal host, and the Apple GPU
compute graph. It calls the system Metal API through `objc2-metal`;
the MSL source is embedded in the executable, compiled at process startup with
fast math disabled, and turned into compute pipeline states owned by the
backend thread. `marian-cpu` owns the portable FP32 Transformer/SSRU executor,
the complete Q8 Transformer/SSRU executor, pure-Rust CPU kernels, and
tokenizer-aware long-text segmentation. Q8 dense weights remain quantized;
embedding rows and the final shortlist are materialized only as needed.

There is no MLX library, CXX inference bridge, native tokenizer, or external
CPU inference process. Both the native Metal and portable CPU hosts are Rust.

`--backend cpu` selects Q8 or FP32 from the validated model manifest. It is the
automatic Linux and container backend. `--cpu-threads` accepts 1, 2, or 4 and
sets both `MATMUL_NUM_THREADS` and `RAYON_NUM_THREADS` before inference starts.
It controls FP32 matrix multiplication and Q8 rten/exact-AVX2 row parallelism;
it does not create additional model owners or weight replicas. The executor
enforces a 256-piece source-chunk cap, a bounded generation cap, and a padded
`batch * source_length^2` budget. These are engine-level limits: HTTP timeouts
do not cancel synchronous inference already running on the backend owner
thread.

The Q8 path strictly parses the existing Marian binary v1 artifact and
validates the complete expected tensor set. All 70 quantized weight tensors
were checked byte-for-byte against quantization of the FP32 artifact. The
runtime executes the same six-layer Transformer encoder and four-layer SSRU
decoder as the FP32 path without dequantizing the whole model.

Inputs longer than one CPU chunk pass through the pure-Rust segmenter. It uses
the real source tokenizer to verify piece counts, prefers sentence boundaries,
falls back safely for punctuation-free input, preserves whitespace and
newlines, and reassembles outputs in original order. Chunk sub-batches remain
within the same quadratic-work budget.

## Concurrency model

Tokio handles many connections, but inference state is deliberately owned by
one OS thread per loaded model. This avoids unsafe cross-thread GPU objects and
uncontrolled weight duplication. Requests enter a bounded queue and compatible
items are collected for a short window. Batch membership is based on direction,
maximum output length, and a power-of-two source-length bucket.

Important invariants:

- queue capacity is finite;
- overload is visible to the caller;
- model construction, inference, and destruction happen on the owner thread;
- a production backend never silently falls back to echo;
- output order matches input order for `/imme` batches;
- readiness turns false before shutdown drains the worker.

## Native Metal backend

The backend implements the graph used by the current English-to-Chinese
Mozilla `base-memory` model: a six-layer Transformer encoder, a four-layer
SSRU decoder, greedy beam-1 decoding, and a lexical shortlist. The Rust host
memory-maps and validates FP32 safetensors, records direct Metal compute
commands, and keeps model buffers, a reusable buffer arena, decoder state, and
pipeline states on its owner thread. The manifest and artifact checksums are
validated before loading. FP32 model storage is the default; the explicit
`MARIAN_MLX_METAL_PRECISION=mixed-f16` mode converts uploaded model storage to
FP16 while leaving activations and reductions in FP32, and reports that mode
through `/info`.
Supported self-attention and single-query cross-attention shapes use a fused
four-query Metal kernel. It streams 32-key tiles, updates an online softmax,
and writes only the final value accumulation rather than an O(N^2) score
matrix. `MARIAN_MLX_METAL_ATTENTION=classic` retains the three-kernel score,
softmax, and value path for compatibility and controlled A/B measurements;
unsupported head dimensions fall back to it. Attention mode is reported by
`/info`.

Embedding and each encoder layer remain separately submitted to bound all
other arena lifetimes and to keep the classic fallback's worst-case score
allocation scoped to one layer. Do not merge those submissions without
measuring worst-case source-length memory. A command-queue creation or
execution failure also makes the backend not-ready so the scheduler stops
admitting work to a failed device.

The public backend and Cargo feature are named `metal`. `mlx` remains only as a
compatibility alias and selects the same implementation. A release does not
need `libmlx.dylib` or an external `.metallib`: runtime-compiled MSL is embedded
in the server executable. Use Instruments Metal System Trace to validate GPU
execution and profile command-buffer or kernel behavior.

## Adding a language direction

1. Confirm the upstream model's architecture, vocabulary, shortlist, beam,
   source, and license.
2. Extend the manifest schema only when the graph actually differs.
3. Add conversion-time shape and checksum validation.
4. Add a golden corpus against a trusted reference backend.
5. Add the direction to the runtime registry and scheduler key.
6. Document quality and memory separately from throughput.

List a direction only after its runtime and tests are in place.

## Release boundary

Source, runtime binaries, and model files are tracked separately. Native macOS
release archives contain one server executable; model files remain separate
operator downloads. The Rust host and project MSL kernels are MIT.
Pure-Rust SentencePiece inference comes from the Apache-2.0
`sentencepiece-rust` crate. The Linux image contains the Rust server and does
not contain a model; the operator downloads verified model artifacts into a
separate volume.
