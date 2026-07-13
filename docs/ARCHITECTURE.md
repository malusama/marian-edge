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
    +--> MLX lazy graph --> Metal command queue (native macOS)
    |
    +--> persistent Bergamot worker / Ruy CPU (Linux image)
```

`marian-server` owns HTTP contracts, input limits, CORS, health endpoints, and
shutdown. `marian-core` owns backend-neutral scheduling and metrics. `marian-mlx`
owns model validation, tokenization, the CXX bridge, and the Apple GPU graph.
The CPU backend uses the same `TranslationBackend` trait and talks to one
persistent worker over a length-prefixed pipe protocol.

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

## Native MLX backend

The decoder implements the graph used by the current English-to-Chinese
Mozilla `base-memory` model: a six-layer Transformer encoder, a four-layer
SSRU decoder, greedy beam-1 decoding, and a lexical shortlist. The manifest is
validated before loading. FP32 is the supported default because it currently
has the best measured parity/performance combination on the M1 baseline.

MLX lazy evaluation means a Rust function call is not proof of GPU execution.
Use Instruments Metal System Trace for performance validation.

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

Source, runtime binaries, and model files are tracked separately.
The repository and native runtime are MIT; MLX remains under its own MIT
license; the operator downloads model artifacts directly from Mozilla. Docker
uses a pinned MPL-2.0 Bergamot source revision and does not contain a model.
