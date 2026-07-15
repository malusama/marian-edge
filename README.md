# Marian MLX

[![CI](https://github.com/malusama/marian-mlx/actions/workflows/ci.yml/badge.svg)](https://github.com/malusama/marian-mlx/actions/workflows/ci.yml)
[![MIT](https://img.shields.io/badge/service-MIT-blue.svg)](LICENSE)

Local English-to-Chinese translation with a Rust HTTP service. On Apple
Silicon, a Rust inference host drives Metal directly through `objc2-metal` and
runtime-compiles embedded MSL compute kernels, including fused
FlashAttention-style online softmax. It does not link MLX or use a C++ inference
bridge. Linux and other portable builds use a pure-Rust CPU
engine with complete Q8 and FP32 Transformer/SSRU graphs, a lexical shortlist,
and greedy decoding.

The repository name is retained for compatibility. Tokenization, long-text
segmentation, scheduling, model loading, and both inference hosts are written
in Rust. The repository no longer contains the former Bergamot/C++ runtime.

[中文说明](README.zh-CN.md)

## Choose the right runtime

| Host | Runtime | Compute | Start command |
|---|---|---|---|
| Apple Silicon Mac, macOS 14+ | native single executable | direct Metal GPU | one-line installer below |
| Linux AMD64 | container | pure-Rust Q8 CPU | `docker compose up -d` |
| Linux ARM64 | container | pure-Rust Q8 CPU | `docker compose up -d` |
| Docker Desktop on a Mac | Linux ARM container | pure-Rust Q8 CPU, **not Metal** | `docker compose up -d` |
| macOS or Linux source build | native executable | pure-Rust Q8 or FP32 CPU | `--features cpu -- --backend cpu` |

Docker cannot pass the macOS Metal device into its Linux VM. Use the native
installer when Apple GPU inference is the goal.

## Native Apple GPU: one command

The installer runs as the current user, verifies downloads, installs a
LaunchAgent on `127.0.0.1:3000`, and converts the model locally.

```sh
curl --proto '=https' --tlsv1.2 -fsSL \
  https://raw.githubusercontent.com/malusama/marian-mlx/main/scripts/install-macos.sh | sh
```

For a reproducible pinned install:

```sh
curl --proto '=https' --tlsv1.2 -fsSL \
  https://raw.githubusercontent.com/malusama/marian-mlx/v0.5.0/scripts/install-macos.sh | \
  MARIAN_MLX_VERSION=v0.5.0 sh
```

`v0.1.1` remains available as the last historical MLX/Bergamot release. Its
runtime layout is not compatible with the direct Metal bundle contract used by
`v0.2.0` and later. Use `v0.2.1` or newer when rollback across those layouts is
required.

You can inspect the script before running it. First install needs about 750 MB
of free space and takes longer because
the model and a pinned Python conversion environment are prepared locally.

```sh
~/.local/bin/marian-mlxctl status
~/.local/bin/marian-mlxctl verify
~/.local/bin/marian-mlxctl logs
~/.local/bin/marian-mlxctl restart
~/.local/bin/marian-mlxctl update
~/.local/bin/marian-mlxctl uninstall          # keeps the model/cache
~/.local/bin/marian-mlxctl uninstall --purge  # removes everything
```

Override the port with `MARIAN_MLX_PORT=3100`. The installer does not take a
port owned by another process and rolls back if `/readyz` fails.

## Docker CPU: one command

Pull explicitly when upgrading so an older local `:cpu` image is not reused:

```sh
docker compose pull
docker compose up -d
docker compose ps
curl -fsS http://127.0.0.1:3000/info
```

Or without Compose:

```sh
docker run -d --name marian-mlx --restart unless-stopped \
  -p 127.0.0.1:3000:3000 \
  -v marian-mlx-models:/models \
  ghcr.io/malusama/marian-mlx:cpu-0.5.0
```

The published image is multi-architecture AMD64/ARM64, non-root, and CPU-only.
It does not embed model bytes: on first start it downloads the pinned `en ->
zh` release directly from Mozilla storage into the named volume and verifies
compressed and uncompressed SHA-256 values. Later starts reuse the volume.

The CPU model has one owner, so changing the compute-thread count does not load
extra model replicas. Concurrent HTTP requests are still micro-batched before
inference. `MARIAN_MLX_CPU_THREADS` accepts 1, 2, or 4 and controls both FP32
matrix multiplication and the Q8 rten/exact-AVX2 row-parallel kernels:

```sh
MARIAN_MLX_CPU_THREADS=2 docker compose up -d --force-recreate
```

The model remains single-owner at every setting. Measure the actual host and
traffic before increasing its internal compute parallelism.

## Immersive Translate

1. Confirm `http://127.0.0.1:3000/readyz` returns HTTP 200.
2. In Immersive Translate, enable beta testing features under **Options >
   Developer settings**.
3. Under **Options > General**, select **Custom API**.
4. Set the API URL to `http://127.0.0.1:3000/imme`.
5. Select English as the source and Simplified Chinese as the target.

The service has CORS disabled by default. Browser extensions with localhost
permission normally do not need it. If the extension reports a CORS error,
re-run the pinned installer with a trusted extension origin. A wildcard is
available for a loopback-only personal deployment:

```sh
curl --proto '=https' --tlsv1.2 -fsSL \
  https://raw.githubusercontent.com/malusama/marian-mlx/v0.5.0/scripts/install-macos.sh | \
  MARIAN_MLX_VERSION=v0.5.0 MARIAN_MLX_CORS_ORIGIN='*' sh
```

For Docker, add `MARIAN_MLX_CORS_ORIGIN: "*"` under `environment` only if required.
See the exact payload and troubleshooting in
[the Immersive Translate guide](docs/IMMERSIVE_TRANSLATE.md).

## API

```sh
curl -fsS http://127.0.0.1:3000/translate \
  -H 'content-type: application/json' \
  -d '{"text":"The weather is beautiful today.","from":"en-US","to":"zh-CN"}'
```

```json
{"text":"...","from":"en","to":"zh"}
```

| Endpoint | Purpose |
|---|---|
| `POST /translate` | one text item |
| `POST /imme` | Immersive Translate-compatible text list |
| `POST /detect` | small English/CJK heuristic, not general language ID |
| `GET /livez` | event-loop liveness |
| `GET /readyz` | model worker ready to accept traffic |
| `GET /health` | compatibility `{"status":"ok"}` response |
| `GET /info` | version, revision, backend, device, precision, attention mode, model, and uptime |
| `GET /metrics` | Prometheus counters and gauges |

Region variants such as `en-US`, `en_US`, `zh-CN`, and `zh-Hans` are normalized
to `en` and `zh`. The current release supports only English to Chinese.
`max_output_tokens` is supported by both the direct Metal and pure-Rust CPU
backends.

## Current scope

| Capability | Status |
|---|---|
| macOS Apple Silicon / direct Metal FP32 | supported |
| macOS Apple Silicon / direct Metal mixed-f16 storage | explicit opt-in; 198/200 exact against FP32 in the qualification corpus |
| Linux AMD64 / pure-Rust Q8 CPU | supported |
| Linux ARM64 / pure-Rust Q8 CPU | supported; tested on ARM64 |
| portable pure-Rust FP32 CPU | supported with an FP32 manifest |
| pure-Rust Q8 Transformer/SSRU graph | supported; dense weights stay quantized |
| pure-Rust SentencePiece and long-text segmentation | supported |
| English-to-Chinese `base-memory` model | supported |
| Transformer encoder + SSRU greedy decoder + shortlist | supported |
| bounded admission and shape-aware micro-batching | supported |
| additional language directions | not yet |
| beam search greater than one | not yet; this release uses beam 1 |
| general language detection | not included |

Each `/imme` list item remains one output item. The Metal source limit is 4,096
tokens; its default fused attention path has linear score-scratch growth. The
CPU engine keeps each inference chunk within 255 source pieces plus EOS and a
bounded padded-attention budget because its encoder attention is quadratic.
Longer text is split at tokenizer-aware sentence boundaries and
reassembled in order while preserving separators, including newlines.

The Q8 backend matches all five release golden translations exactly. On a
200-item differential corpus it matched the retired CPU reference exactly on
164 items; near-tie token choices account for the remaining differences, so
this is not a claim of bit-for-bit output equivalence. Tested repeated
80-sentence input and newline cases matched the retired long-text baseline.

## Architecture

```text
many HTTP requests
        |
        v
 Axum / Tokio validation
        |
        v
 bounded admission queue -- full --> 503 + Retry-After
        |
        v
 direction + shape micro-batcher
        |
        v
 one backend-owner OS thread
        |
        +--> Rust host --> embedded MSL --> Metal GPU (native macOS)
        |
        +--> pure Rust Transformer/SSRU --> Q8/FP32 CPU (portable)
```

Backend state stays on one owner thread. CPU dense operations retain Q8 weights
or use FP32 weights according to the model manifest. See [architecture and
maintenance](docs/ARCHITECTURE.md).
Measured M1-first follow-up work and hardware-validation boundaries are recorded in the
[optimization roadmap](docs/OPTIMIZATION_ROADMAP.md).

## Build from source

Portable service/API checks need only Rust:

```sh
make check
cargo run -p marian-server -- --backend echo
```

The echo backend is development-only and is never an automatic fallback.

The portable pure-Rust CPU backend selects Q8 or FP32 from the model manifest.
Linux `auto` and the published `:cpu` image use this backend; the native Metal
path uses the converted FP32 model.

```sh
scripts/prepare-enzh-model.sh
cargo build --locked --release -p marian-server --features cpu
target/release/marian-mlx-server --backend cpu --cpu-threads 4 \
  --model-dir models/enzh
```

`--backend cpu-q8`, `--backend cpu-fp32`, and `--backend rust` are compatibility
aliases for `cpu`; the manifest still determines Q8 versus FP32. The compute
thread count is fixed before inference starts.

Native Apple GPU prerequisites are the macOS SDK/Command Line Tools, Rust 1.86,
and `uv`:

```sh
scripts/prepare-enzh-model.sh
cargo build --locked --release -p marian-server --features metal
target/release/marian-mlx-server --backend metal --model-dir models/enzh
```

FP32 is the default Metal precision contract. An explicit mixed-precision
storage mode converts model weights to FP16 on upload while retaining FP32
activations and reductions:

```sh
MARIAN_MLX_METAL_PRECISION=mixed-f16 \
  target/release/marian-mlx-server --backend metal --model-dir models/enzh
```

The opt-in mode reports `mixed-f16` from `/info`; it does not silently replace
FP32. It matched 198/200 translations exactly in the deterministic CPU-FP32
versus Metal corpus, so deployments that require the FP32 token contract must
leave the variable unset.

Metal attention defaults to the fused four-query path for the current model's
encoder self-attention and decoder cross-attention. It streams 32-key tiles,
uses online softmax, and never writes an O(N^2) score matrix. `/info` reports
`flash-q4-auto@1`. Keep `auto` in production; `classic` and forced `flash` are
available for qualification and A/B measurements:

```sh
MARIAN_MLX_METAL_ATTENTION=classic \
  target/release/marian-mlx-server --backend metal --model-dir models/enzh

MARIAN_MLX_METAL_ATTENTION=auto \
MARIAN_MLX_METAL_FLASH_THRESHOLD=1 \
  target/release/marian-mlx-server --backend metal --model-dir models/enzh
```

On the measured Apple M1 / 16 GB host, the deployment throughput knee is
`--max-batch-size 16 --batch-window-us 750` with about 32 concurrent short
requests. Concurrency 64 produced essentially no additional throughput and
roughly doubled median latency. The M1-qualified duplicate-row width defaults
to 9 and can be overridden with
`MARIAN_MLX_METAL_DUPLICATE_BATCH_WIDTH`; this coalesces exact duplicates only
inside the current dynamic batch and does not cache results. Use FP32 for the
exact qualified output contract. `mixed-f16` is the memory-first option from
the earlier precision qualification and differed on 2/200 corpus outputs.

`mlx` remains accepted as a feature and backend alias for existing automation,
but it selects the same direct Metal implementation. MSL source is embedded in
the executable and compiled through the Metal framework at process startup, so
there is no `libmlx.dylib`, external `.metallib`, MLX submodule, or
`scripts/build-mlx.sh` step. Native releases ship one executable; the model
directory remains a separately downloaded operator artifact.

The model preparation script pins Python and converter package versions and
verifies model artifacts with SHA-256. Models, converted weights, caches, and
build output stay out of Git.

## Performance

The direct Metal backend has a commit-pinned M1 result, 200-item parity report,
peak-memory measurements, and Instruments trace. Under the same formal release
settings, v0.5.0 reached three-run medians of 599.32 item/s for the repeated
short request and 165.29 item/s for the 200-item corpus: 11.7% and 35.4% above
v0.1.0 MLX FP32. Metal FP32 matched CPU FP32 on all 200 deterministic items.
These are engineering measurements on one Apple M1, not M2-M4 claims. Exact
commands, model and corpus hashes, latency distributions, Q8 allocation
results, and the retired MLX baseline are in [the benchmark
notes](docs/BENCHMARKS.md).

## Security, maintenance, and licensing

The listener is loopback-only in the supported deployment examples. The server
has no authentication or TLS and must not be exposed directly to an untrusted
network. Operational commands and health semantics are in
[the operations guide](docs/OPERATIONS.md); contribution rules are in
[CONTRIBUTING.md](CONTRIBUTING.md); private reports follow
[SECURITY.md](SECURITY.md).

The Rust service and project MSL kernels are MIT licensed. Pure-Rust
SentencePiece inference is provided by the Apache-2.0 `sentencepiece-rust`
crate. CPU kernels use Rust crates recorded in `Cargo.lock`. Model files are
not redistributed by this project; operator-triggered scripts fetch them from
the upstream registry. See [third-party notices](THIRD_PARTY_NOTICES.md).

Firefox and Mozilla are trademarks of the Mozilla Foundation in the United
States and other countries.
