# Marian MLX

[![CI](https://github.com/malusama/marian-mlx/actions/workflows/ci.yml/badge.svg)](https://github.com/malusama/marian-mlx/actions/workflows/ci.yml)
[![MIT](https://img.shields.io/badge/service-MIT-blue.svg)](LICENSE)

Local English-to-Chinese translation with a Rust HTTP service. The
native Apple Silicon runtime executes the Marian model with MLX on the Metal
GPU. The portable Linux image uses the official Bergamot runtime on CPU,
including native Ruy/NEON support on ARM64.

[中文说明](README.zh-CN.md)

## Choose the right runtime

| Host | Runtime | Compute | Start command |
|---|---|---|---|
| Apple Silicon Mac, macOS 14+ | native bundle | MLX / Metal GPU | one-line installer below |
| Linux AMD64 | container | Bergamot / CPU | `docker compose up -d` |
| Linux ARM64 | container | Bergamot / Ruy + NEON CPU | `docker compose up -d` |
| Docker Desktop on a Mac | Linux ARM container | CPU, **not Metal** | `docker compose up -d` |

Docker cannot pass the macOS Metal device into its Linux VM. Use the native
installer when Apple GPU inference is the goal.

## Native Apple GPU: one command

The installer runs as the current user, verifies downloads, and installs a
LaunchAgent on `127.0.0.1:3000`. Model conversion also runs locally.

```sh
curl --proto '=https' --tlsv1.2 -fsSL \
  https://raw.githubusercontent.com/malusama/marian-mlx/main/scripts/install-macos.sh | sh
```

For a pinned release:

```sh
curl --proto '=https' --tlsv1.2 -fsSL \
  https://raw.githubusercontent.com/malusama/marian-mlx/v0.1.1/scripts/install-macos.sh | \
  MARIAN_MLX_VERSION=v0.1.1 sh
```

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

```sh
docker compose up -d
docker compose ps
curl -fsS http://127.0.0.1:3000/info
```

Or without Compose:

```sh
docker run -d --name marian-mlx --restart unless-stopped \
  -p 127.0.0.1:3000:3000 \
  -v marian-mlx-models:/models \
  ghcr.io/malusama/marian-mlx:cpu
```

The image is multi-architecture and runs as a non-root user. It does not embed
model bytes: on first start it downloads the pinned `en -> zh` release directly
from Mozilla storage into the named volume and verifies compressed and
uncompressed SHA-256 values. Later starts reuse the volume.

CPU translation uses one worker by default. One worker still batches concurrent
HTTP requests and kept peak memory near 0.4-0.5 GB in our ARM64 model smoke
test. If the host has spare memory and sustained parallel traffic, try two
workers and measure again; the same test reached about 0.7-0.8 GB because each
active worker owns another model workspace.

```sh
MARIAN_MLX_CPU_THREADS=2 docker compose up -d
```

Start with one worker on small ARM devices, NAS hosts, and Docker Desktop. More
workers are a throughput-versus-memory setting, not a generally faster default.

## Immersive Translate

1. Confirm `http://127.0.0.1:3000/readyz` returns HTTP 200.
2. In Immersive Translate, enable beta testing features under **Options >
   Developer settings**.
3. Under **Options > General**, select **Custom API**.
4. Set the API URL to `http://127.0.0.1:3000/imme`.
5. Select English as the source and Simplified Chinese as the target.

The service has CORS disabled by default. Browser extensions with localhost
permission normally do not need it. If the extension reports a CORS error,
re-run the native installer explicitly with a trusted extension origin. A
wildcard is available for a loopback-only personal deployment:

```sh
curl --proto '=https' --tlsv1.2 -fsSL \
  https://raw.githubusercontent.com/malusama/marian-mlx/main/scripts/install-macos.sh | \
  MARIAN_MLX_CORS_ORIGIN='*' sh
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
| `GET /info` | version, revision, backend, device, precision, model, and uptime |
| `GET /metrics` | Prometheus counters and gauges |

Region variants such as `en-US`, `en_US`, `zh-CN`, and `zh-Hans` are normalized
to `en` and `zh`. The current release supports only English to Chinese.
`max_output_tokens` is supported by MLX; Bergamot uses the model's fixed
`max-length-factor` and rejects non-default values.

## Current scope

| Capability | Status |
|---|---|
| macOS Apple Silicon / MLX v0.32 / Metal | supported; checked with a Metal trace |
| Linux AMD64 / Bergamot int8 CPU | supported |
| Linux ARM64 / Bergamot Ruy + NEON CPU | supported; tested on ARM64 |
| English-to-Chinese `base-memory` model | supported |
| Transformer encoder + SSRU greedy decoder + shortlist | supported |
| bounded admission and shape-aware micro-batching | supported |
| additional language directions | not yet |
| beam search greater than one | not yet; this release uses beam 1 |
| general language detection | not included |

Each `/imme` list item is one input sequence. The MLX source limit is 4,096
tokens; paragraph-level sentence splitting is not yet implemented.

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
        +--> MLX graph --> Metal command queue       (native macOS)
        |
        +--> persistent Bergamot worker --> Ruy CPU  (Linux container)
```

Backend state stays on one owner thread. The Bergamot process uses a small
length-prefixed protocol and is reused across requests. See
[architecture and maintenance](docs/ARCHITECTURE.md).

## Build from source

Portable service/API checks need only Rust:

```sh
make check
cargo run -p marian-server -- --backend echo
```

The echo backend is development-only and is never an automatic fallback.

Native Apple GPU prerequisites are Xcode with the Metal toolchain, CMake 3.25+,
Rust 1.86, and `uv`:

```sh
xcodebuild -downloadComponent MetalToolchain
git submodule update --init --recursive
scripts/build-mlx.sh
scripts/prepare-enzh-model.sh
scripts/build-release.sh
MARIAN_MLX_METALLIB="$PWD/build/mlx-install/lib/mlx.metallib" \
  target/release/marian-mlx-server --backend mlx --model-dir models/enzh
```

The scripts pin Python and converter package versions, and verify the MLX CMake
dependencies and model artifacts with SHA-256. Models, converted weights,
caches, and build output stay out of Git.

## Performance

On the recorded M1 short-sentence workload, the FP32 MLX runtime reached
536.04 requests/s at concurrency 32 versus 95.61 requests/s for the default
one-worker Bergamot int8 container. Instruments captured Metal command
execution. This is one machine and one request shape, not a universal claim; see
[the full methodology](docs/BENCHMARKS.md).

## Security, maintenance, and licensing

The listener is loopback-only in the supported deployment examples. The server
has no authentication or TLS and must not be exposed directly to an untrusted
network. Operational commands and health semantics are in
[the operations guide](docs/OPERATIONS.md); contribution rules are in
[CONTRIBUTING.md](CONTRIBUTING.md); private reports follow
[SECURITY.md](SECURITY.md).

The service is MIT licensed. MLX is MIT. The Docker CPU backend uses
the official MPL-2.0 Bergamot source at a pinned revision. Model files are not
redistributed by this project; operator-triggered scripts fetch them from the
upstream registry. See [third-party notices](THIRD_PARTY_NOTICES.md).

Firefox and Mozilla are trademarks of the Mozilla Foundation in the United
States and other countries.
