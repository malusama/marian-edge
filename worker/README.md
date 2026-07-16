# Cloudflare Worker experiment

This directory packages the English-to-Chinese Q8 backend as a raw SIMD128
WebAssembly module and a small Cloudflare Worker shell. Model artifacts stay in
a private R2 bucket and are loaded once per isolate.

This is an experimental deployment, not the default production target. The
Worker-specific packed ABI uses about 84.94 MiB of Wasm linear memory, down
from 114.19 MiB for the canonical Q8 loader. The endpoint requires a bearer
token but does not yet implement per-customer quotas or rate limiting.

## Build and smoke test

Install the `wasm32-unknown-unknown` Rust target, then run from this directory:

```sh
npm run build:wasm
npm run pack:model
npm run smoke:packed
```

The scripts expect the verified canonical Q8 artifacts under
`/tmp/marian-worker-model`. `pack:model` runs the converter inside the actual
SIMD128 Wasm build and emits `model.worker-packed-v2.bin` plus
`manifest.worker.json`. The smoke test splits the bundle into final ownership
sections and checks the five release golden translations.

## R2 layout

The `MODELS` binding points at `marian-edge-models`, with these private keys:

```text
enzh-q8-packed-v3/model.worker-bundle-v3.bin
```

The v3 transport bundle is pinned by size and R2 ETag and groups the manifest,
tokenizers, shortlist, and packed metadata into one bootstrap section. It loads
with four R2 ranges instead of the previous ten serial metadata/head/range
operations, while still avoiding a transient 49 MB JS allocation. Dense
matrices share one aligned `Vec<u32>` and both embeddings take ownership of
their final byte buffers. The runtime performs no canonical tensor parse,
dense repack, or large copy.

The artifact ABI is deliberately tied to `rten-gemm 0.21.0` and kernel
`wasm-u8i8i32`. Kernel, version, architecture, tensor order, dimensions, packed
offsets, and expected packed sizes are all validated before inference.

## API

```sh
curl https://marian-worker.malu.moe/v1/translate \
  -H 'content-type: application/json' \
  -H "authorization: Bearer $MARIAN_WORKER_TOKEN" \
  --data '{"text":"The weather is beautiful today.","source":"en","target":"zh"}'
```

Only `en -> zh` and at most 128 output tokens per input are supported. Send
`{"texts":[...]}` with 1-16 strings to execute a real model batch; the original
single-string `{"text":"..."}` contract remains unchanged.

`API_TOKEN` must be configured as a Wrangler secret. The local test deployment
keeps its generated token in `/tmp/marian-worker-api-token` with mode `0600`.

## Measured TPE cost inputs

On 2026-07-16, the original canonical loader consumed approximately:

| Input characters | Input tokens | Output tokens | Worker CPU ms |
| ---: | ---: | ---: | ---: |
| 31 | 7 | 5 | 75 |
| 95 | 21 | 15 | 271 |
| 191 | 42 | 30 | 449 |
| 319 | 70 | 50 | 717 |

That is roughly 2.3-2.8 CPU ms per input character for this repetitive sample.
With the packed v2 loader, a live TPE cold event measured 274 CPU ms / 2.11 s
wall and warm golden requests measured 49-60 CPU ms. Cost projections must
include the expected cold-start ratio and average characters per request;
request count alone is insufficient.

## Optimized v3 measurements

On 2026-07-16 the relaxed-SIMD 4x16 kernel, SIMD shortlist/elementwise paths,
real batch API, and four-range v3 loader were deployed as Worker version
`d914a09b-50e3-4cd3-8d66-4537a5b66995`. The 200-item corpus run from this M1
Pro through TPE measured:

| Workload | Local Metal texts/s | Worker texts/s | Worker p50 | Worker p95 |
| --- | ---: | ---: | ---: | ---: |
| Single text, concurrency 1 | 25.75 | 2.81 | 252 ms | 606 ms |
| Single text, concurrency 8 | 68.42 | 20.87 | 245 ms | 714 ms |
| Explicit batch 4, concurrency 1 | 72.81 | 6.06 | 405 ms/request | 1,198 ms/request |
| Explicit batch 4, concurrency 4 | 112.73 | 22.88 | 380 ms/request | 2,322 ms/request |
| Explicit batch 8, concurrency 1 | 117.48 | 11.40 | 571 ms/request | 1,242 ms/request |

Live Tail separated Worker compute from network latency. Six warm single-text
events used 86-112 CPU ms (100.3 ms mean); four warm batch-4 events used
171-195 CPU ms (187.5 ms mean, or 46.9 ms/text). Compared with the previous
same-corpus Worker run, concurrency-8 throughput rose from 15.51 to 20.87
texts/s. A sampled v3 cold event used 383 CPU ms / 1.47 s wall. These are
deployment observations, not a fixed CPU-SKU guarantee.
