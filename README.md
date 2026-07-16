# Marian Edge

[![CI](https://github.com/malusama/marian-edge/actions/workflows/ci.yml/badge.svg)](https://github.com/malusama/marian-edge/actions/workflows/ci.yml)
[![MIT](https://img.shields.io/badge/service-MIT-blue.svg)](LICENSE)

Marian Edge is a local English-to-Chinese HTTP service:

- Apple Silicon Macs use a native Metal backend;
- Linux AMD64 and ARM64 use a pure-Rust Q8 CPU backend;
- models, tokenization, long-text splitting, and request scheduling stay local.

The current release supports English to Chinese and uses greedy decoding with
`beam=1`. Old `marian-mlx` commands, environment variables, and model-format
names remain only as migration aliases.

[中文说明](README.zh-CN.md)

## Choose a runtime

| Host | Recommended setup | Compute |
|---|---|---|
| Apple Silicon Mac, macOS 14+ | native installer | Metal GPU |
| Linux AMD64/ARM64 | Docker Compose | Q8 CPU |
| Docker Desktop on a Mac | Docker Compose | Linux ARM CPU, not Metal |
| macOS/Linux development | source build | Metal or CPU |

Docker Desktop cannot pass the macOS Metal device into its Linux VM. Use the
native installer for GPU inference on a Mac.

## Native macOS install

```sh
curl --proto '=https' --tlsv1.2 -fsSL \
  https://raw.githubusercontent.com/malusama/marian-edge/main/scripts/install-macos.sh | sh
```

Pinned v0.7.0 install:

```sh
curl --proto '=https' --tlsv1.2 -fsSL \
  https://raw.githubusercontent.com/malusama/marian-edge/v0.7.0/scripts/install-macos.sh | \
  MARIAN_EDGE_VERSION=v0.7.0 sh
```

The installer runs without root, verifies downloads, converts the model
locally, and registers a per-user LaunchAgent. Allow at least 750 MB of free
space; the first install also needs time to download the model.

Common commands:

```sh
~/.local/bin/marian-edgectl status
~/.local/bin/marian-edgectl verify
~/.local/bin/marian-edgectl logs
~/.local/bin/marian-edgectl update
~/.local/bin/marian-edgectl rollback
~/.local/bin/marian-edgectl uninstall          # keep model and cache
~/.local/bin/marian-edgectl uninstall --purge  # remove model and cache
```

See the [operations guide](docs/OPERATIONS.md) for start, stop, and diagnostic
commands.

The default listener is `127.0.0.1:3000`. To use 3100, set it during install;
later updates retain the saved port:

```sh
PORT=3100
curl --proto '=https' --tlsv1.2 -fsSL \
  https://raw.githubusercontent.com/malusama/marian-edge/v0.7.0/scripts/install-macos.sh | \
  MARIAN_EDGE_VERSION=v0.7.0 MARIAN_EDGE_PORT="$PORT" sh

SERVICE_ORIGIN="http://127.0.0.1:$PORT"
curl -fsS "$SERVICE_ORIGIN/readyz"
curl -fsS "$SERVICE_ORIGIN/info"
```

## Docker CPU

```sh
docker compose pull
docker compose up -d
docker compose ps
```

Compose maps container port 3000 to host `127.0.0.1:3000` by default. To use
host port 3100, leave the container port unchanged:

```sh
MARIAN_EDGE_HOST_PORT=3100 docker compose up -d
SERVICE_ORIGIN=http://127.0.0.1:3100
curl -fsS "$SERVICE_ORIGIN/readyz"
curl -fsS "$SERVICE_ORIGIN/info"
```

Without Compose:

```sh
docker run -d --name marian-edge --restart unless-stopped \
  -p 127.0.0.1:3000:3000 \
  -v marian-edge-models:/models \
  --read-only --tmpfs /tmp:size=64m,mode=1777 \
  --cap-drop ALL --security-opt no-new-privileges \
  ghcr.io/malusama/marian-edge:cpu-0.7.0
```

The image does not contain model files. On first start it downloads and
verifies the pinned Mozilla English-to-Chinese model, then reuses it from the
Docker volume. `MARIAN_EDGE_CPU_THREADS` accepts `1`, `2`, or `4`; benchmark the
actual host and workload before changing it.

## One port for every endpoint

Marian Edge has one HTTP listener. `/readyz`, `/info`, `/translate`, and
`/imme` must use the same host and port. There is no separate Immersive
Translate port.

| Backend origin | Immersive Translate API URL |
|---|---|
| `http://127.0.0.1:3000` | `http://127.0.0.1:3000/imme` |
| `http://127.0.0.1:3100` | `http://127.0.0.1:3100/imme` |

For example, if a source build starts with `--bind 127.0.0.1:3100`, enter
`http://127.0.0.1:3100/imme` in the extension. Do not copy the default 3000 URL.

The remaining examples use `SERVICE_ORIGIN`. Set it to the address of the
backend that is actually running:

```sh
SERVICE_ORIGIN=http://127.0.0.1:3100  # replace with the actual listener
curl -fsS "$SERVICE_ORIGIN/readyz"
curl -fsS "$SERVICE_ORIGIN/info"
```

## Immersive Translate

1. Confirm the service with the `/readyz` command above.
2. Enable beta testing features under **Options > Developer settings**.
3. Under **Options > General**, select **Custom API**.
4. Set the API URL to **`SERVICE_ORIGIN` followed by `/imme`**. If the backend
   is on 3100, enter `http://127.0.0.1:3100/imme`.
5. Select English as the source and Simplified Chinese as the target.

Browser extensions can usually reach a loopback service without CORS. Configure
`MARIAN_EDGE_CORS_ORIGIN` only if the extension reports a CORS error, and do not
expose the service publicly as a workaround. Payloads and troubleshooting are
covered in the [Immersive Translate guide](docs/IMMERSIVE_TRANSLATE.md).

## API

```sh
SERVICE_ORIGIN=${SERVICE_ORIGIN:-http://127.0.0.1:3000}
curl -fsS "$SERVICE_ORIGIN/translate" \
  -H 'content-type: application/json' \
  -d '{"text":"The weather is beautiful today.","from":"en-US","to":"zh-CN"}'
```

```json
{"text":"...","from":"en","to":"zh"}
```

| Endpoint | Purpose |
|---|---|
| `POST /translate` | translate one text item |
| `POST /imme` | Immersive Translate batch payload |
| `POST /detect` | small English/CJK heuristic |
| `GET /livez` | process liveness |
| `GET /readyz` | model readiness |
| `GET /health` | legacy client compatibility |
| `GET /info` | version, backend, device, and model details |
| `GET /metrics` | Prometheus metrics |

`/translate` accepts `text`, `from`, `to`, and optional `max_output_tokens`; the
default output budget is 512 tokens. `/imme` accepts `source_lang`,
`target_lang`, and `text_list`, with at most 256 items. The complete JSON
request body is limited to 64 KiB. See the [Immersive Translate
guide](docs/IMMERSIVE_TRANSLATE.md) and [operations guide](docs/OPERATIONS.md)
for full fields and error responses.

## Current scope

- Apple Silicon direct Metal FP32 and optional mixed-F16 weight storage.
- Linux AMD64/ARM64 Q8 CPU and pure-Rust CPU with an FP32 manifest.
- SentencePiece, long-text splitting, lexical shortlist, and dynamic batching.
- English-to-Chinese only; `/detect` is not general language identification.
- The decoder is fixed at greedy `beam=1`; beam search is not implemented yet.

The beam-search evaluation plan is in the [optimization
roadmap](docs/OPTIMIZATION_ROADMAP.md). Historical results and test conditions
are in the [benchmark notes](docs/BENCHMARKS.md).

## Build from source

HTTP-layer checks only:

```sh
make check
cargo run -p marian-server -- --backend echo
```

CPU:

```sh
scripts/prepare-enzh-model.sh
cargo build --locked --release -p marian-server --features cpu
target/release/marian-edge-server --backend cpu --cpu-threads 4 \
  --model-dir models/enzh
```

Apple Silicon Metal:

```sh
scripts/prepare-enzh-model.sh
cargo build --locked --release -p marian-server --features metal
target/release/marian-edge-server --backend metal --model-dir models/enzh
```

See [architecture](docs/ARCHITECTURE.md) for code boundaries,
[operations](docs/OPERATIONS.md) for settings, and
[CONTRIBUTING](CONTRIBUTING.md) for the release process.

## Security and licensing

Examples bind only to loopback. The service has no authentication or TLS and
must not be exposed directly to an untrusted network. Report security issues
privately as described in [SECURITY](SECURITY.md).

Service code and project MSL kernels are MIT licensed. Model files are not
distributed in the repository or image; download scripts fetch and verify them
from upstream. See [third-party notices](THIRD_PARTY_NOTICES.md).
