# Cloudflare Worker experiment

This directory packages the English-to-Chinese Q8 backend as a raw SIMD128
WebAssembly module and a small Cloudflare Worker shell. Model artifacts stay in
a private R2 bucket and are loaded once per isolate.

This is an experimental deployment, not the default production target. The
current model uses about 114.2 MiB of Wasm linear memory against Cloudflare's
128 MB per-isolate limit. It translated successfully in TPE, but one resource
limit (1104) response occurred during repeated cold/warm testing. The endpoint
also has no authentication or rate limiting yet.

## Build and smoke test

Install the `wasm32-unknown-unknown` Rust target, then run from this directory:

```sh
npm run build:wasm
node scripts/smoke.mjs
```

The smoke script expects the verified Q8 artifacts under
`/tmp/marian-worker-model`. It checks the five release golden translations.

## R2 layout

The `MODELS` binding points at `marian-edge-models`, with these private keys:

```text
enzh-q8/manifest.json
enzh-q8/model.q8.bin
enzh-q8/source.spm
enzh-q8/target.spm
enzh-q8/shortlist.bin
```

The Worker verifies every object against a build-pinned SHA-256 with Web Crypto
before transferring ownership to Wasm. This avoids spending several seconds of
single-threaded Wasm CPU hashing the 44 MB weight file on every cold isolate.

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

On 2026-07-16, warmed requests consumed approximately:

| Input characters | Input tokens | Output tokens | Worker CPU ms |
| ---: | ---: | ---: | ---: |
| 31 | 7 | 5 | 75 |
| 95 | 21 | 15 | 271 |
| 191 | 42 | 30 | 449 |
| 319 | 70 | 50 | 717 |

That is roughly 2.3-2.8 CPU ms per input character for this repetitive sample.
A cold isolate adds roughly 0.3-0.6 CPU seconds plus 1-3 seconds of R2 and
startup wall time. Cost projections must include the expected cold-start ratio
and average characters per request; request count alone is insufficient.
