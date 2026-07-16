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
enzh-q8-packed-v2/manifest.worker.json
enzh-q8-packed-v2/model.worker-packed-v2.bin
enzh-q8-packed-v2/source.spm
enzh-q8-packed-v2/target.spm
enzh-q8-packed-v2/shortlist.bin
```

Small objects are verified against build-pinned SHA-256 values with Web Crypto.
The 44 MB packed bundle is pinned by size and R2 ETag, then loaded with one head,
one header range, and four section ranges. Dense matrices share one aligned
`Vec<u32>` and both embeddings take ownership of their final byte buffers, so
the runtime performs no canonical tensor parse, dense repack, or large copy.

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

Only `en -> zh`, one input per request, and at most 128 output tokens are
supported in this Worker profile.

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
