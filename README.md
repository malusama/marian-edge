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
  https://raw.githubusercontent.com/malusama/marian-mlx/v0.6.0/scripts/install-macos.sh | \
  MARIAN_MLX_VERSION=v0.6.0 sh
```

`v0.1.1` remains available as the last historical MLX/Bergamot release. Its
runtime layout is not compatible with the direct Metal bundle contract used by
`v0.2.0` and later. Use `v0.2.1` or newer when rollback across those layouts is
required.

You can inspect the script before running it. The installer requires at least
750 MB of free space each time it runs. A first install uses most of that space
and takes longer because the model and a pinned Python conversion environment
are prepared locally.

```sh
~/.local/bin/marian-mlxctl status
~/.local/bin/marian-mlxctl verify
~/.local/bin/marian-mlxctl logs
~/.local/bin/marian-mlxctl restart
~/.local/bin/marian-mlxctl stop
~/.local/bin/marian-mlxctl start
~/.local/bin/marian-mlxctl update
~/.local/bin/marian-mlxctl rollback
~/.local/bin/marian-mlxctl uninstall          # keeps the model/cache
~/.local/bin/marian-mlxctl uninstall --purge  # removes everything
```

To use port 3100, set it on the receiving side of the install pipe and then use
that same port for every endpoint, including Immersive Translate:

```sh
PORT=3100
curl --proto '=https' --tlsv1.2 -fsSL \
  https://raw.githubusercontent.com/malusama/marian-mlx/v0.6.0/scripts/install-macos.sh | \
  MARIAN_MLX_VERSION=v0.6.0 MARIAN_MLX_PORT="$PORT" sh
curl -fsS "http://127.0.0.1:$PORT/readyz"
curl -fsS "http://127.0.0.1:$PORT/info"
# Immersive Translate URL: http://127.0.0.1:3100/imme
```

The saved port is retained by later updates. The installer does not take a
port owned by another process and rolls back if `/readyz` fails.

## Docker CPU: one command

Pull explicitly when upgrading so an older local `:cpu` image is not reused:

```sh
docker compose pull
docker compose up -d
docker compose ps
curl -fsS http://127.0.0.1:3000/info
```

Compose keeps the container service on port 3000. To publish it on host port
3100, change only the host side and use 3100 in every client URL:

```sh
MARIAN_MLX_HOST_PORT=3100 docker compose up -d
curl -fsS http://127.0.0.1:3100/readyz
curl -fsS http://127.0.0.1:3100/info
# Immersive Translate URL: http://127.0.0.1:3100/imme
```

Or without Compose:

```sh
docker run -d --name marian-mlx --restart unless-stopped \
  -p 127.0.0.1:3000:3000 \
  -v marian-mlx-models:/models \
  --read-only --tmpfs /tmp:size=64m,mode=1777 \
  --cap-drop ALL --security-opt no-new-privileges \
  ghcr.io/malusama/marian-mlx:cpu-0.6.0
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

Those URLs assume the default port. If the native installer or Compose example
uses host port 3100, first check `http://127.0.0.1:3100/readyz` and enter
`http://127.0.0.1:3100/imme`; do not mix ports between health checks and the
extension.

The service has CORS disabled by default. Browser extensions with loopback
permission normally do not need it. If the extension reports a CORS error,
re-run the pinned installer with the extension's exact trusted origin. A
wildcard is available only for a loopback-only personal deployment:

```sh
curl --proto '=https' --tlsv1.2 -fsSL \
  https://raw.githubusercontent.com/malusama/marian-mlx/v0.6.0/scripts/install-macos.sh | \
  MARIAN_MLX_VERSION=v0.6.0 MARIAN_MLX_CORS_ORIGIN='*' sh
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
| `GET /readyz` | model worker lifecycle is ready; it does not report spare queue capacity |
| `GET /health` | compatibility `{"status":"ok"}` response |
| `GET /info` | version, revision, backend, device, precision, attention mode, model, and uptime |
| `GET /metrics` | Prometheus counters and gauges |

Region variants such as `en-US`, `en_US`, `zh-CN`, and `zh-Hans` are normalized
to `en` and `zh`. The current release supports only English to Chinese.

| Endpoint | Accepted JSON fields | Limits |
|---|---|---|
| `POST /translate` | `text`, optional `from`, required `to`, optional `max_output_tokens`; `source_lang` and `target_lang` are aliases for `from` and `to` | `max_output_tokens` defaults to 512 and is clamped to 1-2,048 |
| `POST /imme` | optional `source_lang`, required `target_lang`, required `text_list` | at most 256 nonempty items; its contract does not define `max_output_tokens`, so each item starts with the default 512-token budget |
| `POST /detect` | required `text`; returns `{"language":"en"}` or `{"language":"zh"}` | heuristic English/CJK detection only |

Do not mix the two request shapes. The complete JSON request body, including
JSON syntax and all list items, is limited to 64 KiB. Each text item must also
be nonempty. `max_output_tokens` works with both direct Metal and pure-Rust CPU
when using `/translate`; it is a caller ceiling, so EOS and backend
model/runtime limits may stop generation earlier.

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

Each `/imme` list item remains one output item. CPU and Metal use the shared
`marian-core` segmentation policy with exact tokenizer piece counts: each CPU
segment contains at most 255 source pieces plus EOS, and each Metal segment at
most 4,095 source pieces plus EOS. Longer text is split at tokenizer-aware
sentence boundaries and reassembled in order while preserving separators,
including newlines. Automatic segmentation retains one shared
`max_output_tokens` budget for the original item. The HTTP contract separately
limits the complete JSON request body to 64 KiB. CPU also bounds
padded-attention work because its encoder attention is quadratic; fused Metal
attention avoids materializing a quadratic score matrix.

The Q8 backend matches all five release golden translations exactly. On a
200-item differential corpus it matched the retired CPU reference exactly on
164 items; near-tie token choices account for the remaining differences, so
this is not a claim of bit-for-bit output equivalence. This 164/200 comparison
and the repeated 80-sentence/newline checks are historical engineering evidence
whose raw result artifact is not checked into this repository.

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
requests in the v0.6 qualification. A historical v0.4 mixed-f16 sweep found no
further gain at concurrency 64 and substantially higher median latency; that
older point is a sizing warning, not current v0.6 FP32 proof. The M1-qualified
duplicate-row width defaults to 9 and can be overridden with
`MARIAN_MLX_METAL_DUPLICATE_BATCH_WIDTH`. Core always coalesces exact logical
duplicates; this knob rematerializes up to that many physical rows inside the
current dynamic batch for GPU occupancy and never caches results. The remaining
M1 defaults are decode row budget 54, at most six steps per submission, 256
selection threads, and custom FP32 GEMM disabled; `/info` reports the resolved
profile. Use FP32 for the exact qualified output contract. `mixed-f16` is the
memory-first option and differed on 2/200 corpus outputs. M2-M4 profiles are
conservative until measured on those devices.

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

The final v0.6.0 release candidate and a freshly rebuilt v0.1.0 MLX binary were
measured live on the same loaded M1 desktop. The measured candidate reported
0.5.0 because the version bump had not yet been committed; the v0.6.0 release
uses the same inference source. Three-run medians improved from 486.64 to 546.19
item/s for 1,000 short requests (+12.2%) and from 116.68 to 149.14 item/s for
five 200-item corpus requests (+27.8%). Flash q4 was 12.3% and 4.9% faster than
the same final binary's classic attention path, with identical output hashes.
Metal FP32 matched CPU FP32 on all 200 deterministic items; a 300-request Metal
trace completed 40/40 labeled command buffers with zero errors. These are one
Apple M1's engineering measurements, not M2-M4 claims. Exact runs, historical
quiet-host peaks, hashes, latency, memory, trace evidence, and commands are in
[the benchmark notes](docs/BENCHMARKS.md).

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
