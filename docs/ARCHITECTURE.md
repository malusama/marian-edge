# Architecture

This document shows the request path and where each part of the runtime lives.

`marian-core` contains backend-neutral behavior. CPU and Metal crates contain
hardware-specific execution. HTTP routes do not implement batching policy, and
the scheduler does not contain GPU tuning values.

## Request path

```text
HTTP clients
    |
    v
marian-server: validation + protocol adapters
    |
    v
marian-core: bounded admission -- full --> HTTP 503 + Retry-After
    |
    v
canonical ordering + direction/shape micro-batching
    |
    v
one backend-owner OS thread per loaded model
    |
    +--> marian-metal --> embedded MSL / MPS --> Metal GPU
    |
    +--> marian-cpu --> Q8 or FP32 Rust graph --> CPU kernels
```

Tokio owns connection concurrency; it does not own inference state. Model and
device state are constructed, used, and destroyed on one backend thread. This
keeps non-`Send` accelerator objects local, prevents accidental weight replicas,
and makes failure and shutdown state explicit.

## Crate boundaries

| Crate | Owns | Must not own |
|---|---|---|
| `marian-core` | Request/result types, backend trait, bounded admission, canonical batch key, micro-batching, metrics, logical duplicate coalescing, and tokenizer-aware segmentation algorithm | HTTP payloads, model formats, tokenizer implementation, CPU/Metal graph code, or device occupancy constants |
| `marian-model` | Manifest schema, artifact paths, checksums, architecture metadata, and shortlist loading | HTTP or scheduling policy |
| `marian-tokenizer` | Pure-Rust SentencePiece inference boundary | Text-segmentation policy or backend graph code |
| `marian-cpu` | Portable FP32 and Q8 Transformer/SSRU executors, CPU kernels, scratch reuse, and CPU-specific work limits | HTTP contracts or Metal policy |
| `marian-metal` | Metal backend, graph engine, embedded kernels, command submission, workspace/cache lifetimes, and device tuning | HTTP contracts or generic scheduler policy |
| `marian-server` | Axum routes, request/body limits, CORS, health/readiness, backend selection, and protocol compatibility | Length bucketing, duplicate occupancy policy, or inference arithmetic |

`marian-metal` has no production dependency on `marian-cpu`. Its CPU dependency
is dev-only and exists for differential tests. Both production backends depend
on the shared contracts in `marian-core`, `marian-model`, and, when enabled,
`marian-tokenizer`.

Use this placement rule when adding code:

| Change | Destination |
|---|---|
| HTTP validation, payload shape, CORS, or endpoint behavior | `marian-server` |
| Admission, batch compatibility, logical deduplication, or segmentation policy | `marian-core` |
| Manifest, checksum, tensor metadata, or shortlist format | `marian-model` |
| SentencePiece encoding/decoding | `marian-tokenizer` |
| CPU arithmetic, dispatch, representation, or scratch strategy | `marian-cpu` |
| GPU arithmetic, resource lifetime, command grouping, or device tuning | `marian-metal` |

In particular, M-series tile widths, decode submission sizes, and occupancy
floors belong in Metal tuning, never in core scheduling.

The Rust package and source directory are both `marian-metal` and
`crates/marian-metal`. The repository, installed executable, archive, service,
metrics, and canonical `MARIAN_EDGE_*` environment contract use the Marian Edge
name. The historical `MARIAN_MLX_*` runtime settings, `mlx` Cargo feature, and
CLI backend value remain migration aliases; they do not link the MLX runtime.

## Model format

New model manifests use the backend-neutral
`marian-edge.transformer-ssru.v1` namespace. Loaders, the installer, and the
controller continue to accept the historical
`marian-mlx.transformer-ssru.v1` value so existing verified model directories
remain usable. The converter keeps the existing safetensors metadata label and
weight bytes stable; changing the manifest namespace alone must not invalidate
the published FP32 weight checksum.

`marian-model` owns graph schema, architecture validation, the model-position
limit, and the shared sinusoidal table. CPU source, generation, batch, and
padded-attention limits remain private to `marian-cpu`; device tuning remains
private to `marian-metal`.

## Long text

`marian-core::segment_text` is tokenizer-independent: a backend supplies a
closure that returns the exact source-piece count. The shared algorithm prefers
sentence boundaries, has a safe fallback for punctuation-free text, and
preserves separators, whitespace, newlines, and source order.

The execution limit remains backend-specific:

| Backend | Maximum source segment | Per-segment generation ceiling | Additional work bound |
|---|---:|---|---|
| CPU | 255 source pieces plus EOS | minimum of remaining caller budget, `source_tokens * manifest_factor`, and 256 steps | bounded `batch * padded_source_length^2` attention work |
| Metal | 4095 source pieces plus EOS | minimum of remaining caller budget, `source_tokens * manifest_factor`, and 4096 positions | 4096-position model/runtime limit |

Each input still produces one output. Segments are reassembled in order, and
the original input's `max_output_tokens` is one shared budget across all
segments rather than a fresh budget per segment. The HTTP layer separately
limits the complete JSON request body to 64 KiB; `/imme` accepts at most 256
nonempty items, and every text item must be nonempty. Transport limits do not
replace backend token limits. `max_output_tokens` is a caller ceiling, not a
promise to generate that many tokens; EOS and model/runtime ceilings may stop
generation earlier.

## Scheduling

Requests enter a finite admission queue. A batch is compatible when its
language direction, output budget, and power-of-two source-character-count
bucket match. The first incompatible queued item is deferred without draining
the bounded queue into an unbounded shadow queue.

`Translator::translate_many` is the public path for a bounded logical group
such as `/imme`: it sorts by the canonical `TranslationInput::batch_key`,
submits the items, and restores caller order. Protocol code must call it rather
than reproducing bucketing rules.

Within one dynamic batch, core coalesces byte-for-byte identical logical inputs
and calls `translate_batch_with_repetitions` with one row plus its admitted
request count. A backend may use that count to materialize extra physical rows
for an accelerator occupancy knee, but it must return exactly one result per
logical row. The Metal backend makes that physical-row decision from its tuning
profile. There is no cross-batch translation-result cache.

Runtime rules:

- queue capacity is finite and overload is visible to callers;
- model construction, inference, and destruction stay on the owner thread;
- production backends never silently fall back to echo;
- logical output count and caller order are preserved;
- device geometry does not leak into `marian-core`;
- readiness becomes false before shutdown drains or after a terminal backend
  failure.

## CPU backend

`--backend cpu` validates the manifest and selects Q8 or FP32 execution. It is
the automatic Linux and container backend. Q8 dense weights stay quantized;
embedding rows and shortlist values are materialized only when required. The
executor uses reusable scratch and shape-aware kernels. Multiple Q8 workers
share immutable weights, tokenizers, shortlist data, and positional tables
while retaining independent activation scratch and bounded queues.

`--cpu-threads` accepts 1, 2, or 4 and configures both the FP32 matrix-multiply
and Rayon pools before inference starts. `--cpu-workers` accepts 1 through 8;
Q8 workers share model storage, while FP32 workers currently load independent
executors. HTTP timeouts do not cancel synchronous work already executing on an
owner thread.

The Q8 path parses the Marian binary v1 format and validates the expected
tensors. Scalar implementations are used to check hardware-specific kernels
and reductions whose order affects output.

## Native Metal backend

The current English-to-Chinese graph is a six-layer Transformer encoder and a
four-layer SSRU decoder with greedy beam-1 decoding and a lexical shortlist.
The Rust host memory-maps and validates FP32 safetensors, uploads owned model
storage, and records MPS and embedded MSL commands. Fast math is disabled when
the MSL source is compiled at startup.

FP32 is the default precision contract. The explicit
`MARIAN_EDGE_METAL_PRECISION=mixed-f16` mode stores uploaded model weights in
FP16 while retaining FP32 activations and reductions; `/info` reports the
selected precision.

Metal does not currently accept Q8 manifests. M1 exposes fast ARM SDOT to the
CPU backend but no equivalent MPS INT8 matrix-multiply contract; a future Metal
Q8 path needs custom packed-weight/dequantization kernels and separate quality
and performance qualification.

Supported self-attention and single-query cross-attention use a
FlashAttention-style online-softmax kernel. It streams key/value tiles and
writes only the final accumulation instead of an O(N^2) score matrix. Classic
score/softmax/value kernels remain an explicit fallback and A/B path.

The main Metal modules are:

| Module | Responsibility |
|---|---|
| `config` | command-line and environment settings |
| `tuning` | device-family defaults for attention, decode, occupancy, and GEMM |
| `backend` | tokenization, calls into the shared segmenter, physical repetition, and result assembly |
| `engine/model` | model validation and packed weight structures |
| `engine/decode` | candidate preparation, output limits, and the decode loop |
| `engine/ops` | graph operations, Metal parameter values, and dispatch geometry |
| `metal_runtime` | Metal/MPS objects, command buffers, pipelines, and matrix views |
| `workspace` | request and transient buffer arenas |

`MetalBackend::load_with_config` validates the model and calls
`MetalEngine::load`. The engine selects a device and resolves `MetalTuning`.
Decode code calls graph operations, which in turn use `metal_runtime` and
`workspace`; low-level operations do not depend on the decode loop.

### Resource lifetimes

Metal storage is classified by graph lifetime, not by the layer that first
uses it:

| Lifetime | Examples | Reuse rule |
|---|---|---|
| Permanent | Model weights, pipelines, permanent MPS matrix views | Retained for the runtime lifetime |
| Request | Encoded input, encoder output, cross cache, decoder state, request arena | Rewound only when the next inference request begins |
| Transient | Per-command-buffer intermediates and uploads | Eligible for reuse by the next frame after the active frame completes |

`MetalWorkspace` owns the request and transient arenas. `MetalRuntime` mirrors
those lifetimes for MPS matrix views: permanent weight views remain cached,
request-arena views are cleared by `begin_request`, and uncached one-off buffers
never enter either cache. This prevents a cached view from retaining an old
arena buffer across requests while keeping stable model views reusable.

### Tuning boundary

`MetalTuning` is resolved once after device selection and centralizes attention,
GEMM, decode-submission, and duplicate-row policy. Auto detection selects an
M1, M2, M3, M4, or generic profile; `MARIAN_EDGE_METAL_PROFILE` can override it
for controlled qualification. `/info` includes the active profile with the
attention label.

`MARIAN_EDGE_METAL_*` is the current tuning namespace.
`MARIAN_MLX_METAL_*` remains available for migration; setting both forms to
different values is an error.

The M1 defaults are measured in the v0.6.0 release qualification on the
documented Apple M1 host. Later-family and generic defaults are conservative
starting points, not performance claims; they require benchmarks and profiler
evidence on the corresponding hardware. Individual environment overrides
remain available for A/B work.

The Flash encoder and cross-cache build can share one command buffer because
the fused attention path has no quadratic score storage. Classic attention
keeps bounded fallback submissions. Decode submission length is selected from
active rows, remaining output budget, completion state, and the tuning profile;
it is not a hard-coded scheduler concern. Command creation or execution failure
marks the backend not-ready so admission stops.

The public feature and backend name is `metal`; `mlx` remains a compatibility
alias for the same implementation. Releases need neither `libmlx.dylib` nor an
external `.metallib` because the MSL source is embedded in the executable.

## Adding a language direction

1. Confirm the upstream architecture, vocabulary, shortlist, beam, language
   identifiers, and license.
2. Extend the manifest only when the graph contract actually differs.
3. Add conversion-time shape and checksum validation.
4. Add a deterministic golden corpus against a trusted reference backend.
5. Register the direction and verify scheduler-key behavior.
6. Qualify long-text segmentation, output-budget preservation, quality,
   memory, and throughput separately.

List a direction only after its runtime and tests are present.

## Release layout

Source, runtime binaries, and model artifacts are versioned separately. Native
macOS archives contain one server executable; Linux images contain the Rust
server but no model. Operators download verified model artifacts into separate
storage. Downloads and activation remain checksum-verified and atomic.

The Rust host and project MSL kernels are MIT. Pure-Rust SentencePiece inference
comes from the Apache-2.0 `sentencepiece-rust` crate. There is no MLX runtime,
CXX inference bridge, native tokenizer process, or external CPU worker in the
production architecture.
