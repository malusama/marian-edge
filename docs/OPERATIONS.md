# Operations guide

## Native macOS service

The installer creates the per-user LaunchAgent
`io.github.malusama.marian-edge`, binds it to `127.0.0.1:3000` by default, and
stores data under `~/.local/share/marian-edge`. A custom
`MARIAN_EDGE_PORT=3100` is persisted, so operational checks must read the saved
port instead of assuming 3000:

```sh
MARIAN_EDGE_HOME=${MARIAN_EDGE_HOME:-"$HOME/.local/share/marian-edge"}
if [ -r "$MARIAN_EDGE_HOME/config/port" ]; then
  PORT=$(sed -n '1p' "$MARIAN_EDGE_HOME/config/port")
else
  PORT=3000
fi
SERVICE_ORIGIN="http://127.0.0.1:$PORT"

curl -fsS "$SERVICE_ORIGIN/readyz"
curl -fsS "$SERVICE_ORIGIN/info"
```

The native release payload is one server executable. Its MSL source is embedded
and compiled through the system Metal framework when the process starts; there
is no adjacent `libmlx.dylib` or `.metallib` to install or verify. The model
directory is managed separately.

For an upgrade from `marian-mlx`, the installer discovers the historical data
and state directories when the new locations do not exist, treats
`io.github.malusama.marian-mlx` as the previous service, and archives that
LaunchAgent only after Marian Edge passes readiness. The old `MARIAN_MLX_*`
settings and `marian-mlxctl` command remain migration aliases.

### Installed-service lifecycle

`marian-edgectl` is the supported operator interface; a source checkout is not
required:

```sh
~/.local/bin/marian-edgectl status
~/.local/bin/marian-edgectl verify
~/.local/bin/marian-edgectl logs
~/.local/bin/marian-edgectl restart
~/.local/bin/marian-edgectl stop
~/.local/bin/marian-edgectl start
~/.local/bin/marian-edgectl update
~/.local/bin/marian-edgectl rollback
~/.local/bin/marian-edgectl uninstall
~/.local/bin/marian-edgectl uninstall --purge
```

`update` downloads and verifies the next release before stopping the current
service, then gates the cutover on `/readyz`. `rollback` verifies the previous
release, switches `current` and `previous`, rebuilds the LaunchAgent, and waits
for readiness; if that fails it restores the original current release.
`uninstall` keeps the model and cache, while `uninstall --purge` also removes
releases, model, cache, and logs.

If installation used custom locations, export the same values for every
controller invocation and call the controller from that bin directory:

```sh
export MARIAN_EDGE_HOME=/absolute/path/to/marian-edge
export MARIAN_EDGE_STATE=/absolute/path/to/marian-edge-state
export MARIAN_EDGE_BIN_DIR=/absolute/path/to/bin
"$MARIAN_EDGE_BIN_DIR/marian-edgectl" verify
```

The raw `launchctl` equivalents are useful for diagnosis, not normal lifecycle
management:

```sh
launchctl print "gui/$(id -u)/io.github.malusama.marian-edge"
MARIAN_EDGE_STATE=${MARIAN_EDGE_STATE:-"$HOME/.local/state/marian-edge"}
tail -f "$MARIAN_EDGE_STATE/server.log" \
  "$MARIAN_EDGE_STATE/server.error.log"
```

Repository scripts such as `scripts/install-macos.sh` are contributor entry
points when working from a source checkout. Installed users should use
`marian-edgectl`.

### Metal settings and local A/B runs

The installed LaunchAgent does not inherit variables from an interactive
shell. The settings below apply to a separate foreground process; they do not
change an already installed service.

| Environment variable | Default | Values |
|---|---:|---|
| `MARIAN_EDGE_METAL_PRECISION` | `fp32` | `fp32`, `mixed-f16` |
| `MARIAN_EDGE_METAL_PROFILE` | `auto` | `auto`, `m1`, `m2`, `m3`, `m4`, `generic` |
| `MARIAN_EDGE_METAL_ATTENTION` | `auto` | `auto`, `classic`, `flash` |
| `MARIAN_EDGE_METAL_FLASH_THRESHOLD` | `1` | sequence threshold, at most 4096 |
| `MARIAN_EDGE_METAL_FLASH_QUERY_TILE` | profile value | `1`, `2`, `4` |
| `MARIAN_EDGE_METAL_DUPLICATE_BATCH_WIDTH` | profile value | physical row target for duplicate requests |
| `MARIAN_EDGE_METAL_DECODE_ROW_BUDGET` | profile value | rows multiplied by steps per submission |
| `MARIAN_EDGE_METAL_DECODE_MAX_STEPS` | profile value | `1` through `8` |
| `MARIAN_EDGE_METAL_DECODE_SELECTION_THREADS` | profile value | `128`, `256`, `512` |
| `MARIAN_EDGE_METAL_CUSTOM_GEMM_MAX_ROWS` | profile value | `0` disables the custom FP32 GEMM path |

`MARIAN_EDGE_METAL_*` is the current namespace. The old
`MARIAN_MLX_METAL_*` names remain as aliases; startup rejects conflicting
values.

Leave the installed service on its saved port and run an A/B candidate on a
different port:

```sh
MARIAN_EDGE_METAL_ATTENTION=classic \
  target/release/marian-edge-server \
  --backend metal --model-dir models/enzh --bind 127.0.0.1:3101
# In another shell:
curl -fsS http://127.0.0.1:3101/info
```

The M1 defaults and their measurements are in
[BENCHMARKS.md](BENCHMARKS.md). Re-run the sweep before changing another GPU
family's profile.

## Runtime configuration

These variables map to server CLI flags for foreground and container runs.
The native installer persists only its documented install settings such as
port and CORS; shell exports do not alter an already installed LaunchAgent.

| Environment variable | Native default | Meaning |
|---|---|---|
| `MARIAN_EDGE_BIND` | `127.0.0.1:3000` | complete listener address; container image overrides it to `0.0.0.0:3000` |
| `MARIAN_EDGE_BACKEND` | `auto` | `auto`, `metal`, `cpu`, or development-only `echo` |
| `MARIAN_EDGE_MODEL_DIR` | `models/enzh` | model directory; container image uses `/models/en-zh` |
| `MARIAN_EDGE_CPU_THREADS` | `1` | CPU inference threads: `1`, `2`, or `4` |
| `MARIAN_EDGE_QUEUE_CAPACITY` | `256` | bounded admission capacity |
| `MARIAN_EDGE_MAX_BATCH_SIZE` | `16` | maximum logical dynamic batch size |
| `MARIAN_EDGE_MAX_PADDED_SOURCE_CHARS` | `4096` | padded-character work bound for scheduler compatibility |
| `MARIAN_EDGE_BATCH_WINDOW_US` | `750` | micro-batch collection window in microseconds |
| `MARIAN_EDGE_REQUEST_TIMEOUT_MS` | `30000` | end-to-end scheduler timeout in milliseconds |
| `MARIAN_EDGE_CORS_ORIGIN` | unset | one exact origin or `*`; keep unset unless the client requires CORS |
| `MARIAN_EDGE_JSON_LOGS` | `false` | emit JSON rather than text tracing logs |

## Docker CPU service

The container listens on `0.0.0.0:3000` internally. The supported Compose file
publishes it only to host loopback; `MARIAN_EDGE_HOST_PORT` changes the host side
without changing the container port:

```sh
docker compose pull
docker compose up -d
docker compose ps
docker compose logs -f
SERVICE_ORIGIN=http://127.0.0.1:3000
curl -fsS "$SERVICE_ORIGIN/info"
docker compose down
```

Custom host port:

```sh
MARIAN_EDGE_HOST_PORT=3100 docker compose up -d
SERVICE_ORIGIN=http://127.0.0.1:3100
curl -fsS "$SERVICE_ORIGIN/readyz"
curl -fsS "$SERVICE_ORIGIN/info"
# Immersive Translate: $SERVICE_ORIGIN/imme
```

For a reproducible rollback, pin the immutable versioned image. Repeat the same
variables on later Compose commands because shell assignments are not stored:

```sh
MARIAN_EDGE_IMAGE=ghcr.io/malusama/marian-edge:cpu-0.7.0 \
MARIAN_EDGE_HOST_PORT=3100 \
  docker compose pull
MARIAN_EDGE_IMAGE=ghcr.io/malusama/marian-edge:cpu-0.7.0 \
MARIAN_EDGE_HOST_PORT=3100 \
  docker compose up -d
```

The named volume keeps the operator-downloaded model across container updates
and `docker compose down`. Use `docker compose down -v` only when intentionally
removing the model volume too. The image is Linux CPU-only on both Intel and
ARM hosts. Docker Desktop cannot pass the macOS Metal device into a Linux
container; use the native installer for Apple GPU inference.

### CPU ownership and compute threads

The CPU model has one owner. Concurrent HTTP requests still benefit from the
scheduler's micro-batching, and changing `MARIAN_EDGE_CPU_THREADS` does not
create extra model replicas or workers. The setting accepts 1, 2, or 4 and is
applied at startup to FP32 matrix multiplication and Q8 rten/exact-AVX2 row
parallelism.

Raise the value one step at a time, recreate the container, warm the model with
representative traffic, and compare throughput, tail latency, CPU utilization,
and peak RSS:

```sh
MARIAN_EDGE_CPU_THREADS=2 docker compose up -d --force-recreate
docker stats
```

Exact scaling depends on text length, batch shape, architecture, precision,
and model workspace; do not apply sizing numbers from the retired CPU runtime.

## Health checks and errors

| Endpoint | Success means | Suitable for |
|---|---|---|
| `/livez` | HTTP loop is alive | process liveness |
| `/readyz` | model worker lifecycle is ready; queue capacity is separate | traffic readiness |
| `/health` | compatibility endpoint is reachable | existing clients |
| `/info` | backend/device/model identity | deployment verification |
| `/metrics` | Prometheus text is available | capacity monitoring |

`/health` alone does not prove that a model is loaded. Gate traffic on
`/readyz` and inspect `/info` after every deployment. `/readyz` returning 503
means the worker is not ready, draining, or stopped; queue saturation instead
appears on translation requests.

| Status | Meaning | Operator action |
|---:|---|---|
| `400` | request validation failed, for example empty text or too many `/imme` items | correct the request |
| `415` | a JSON endpoint was called without `Content-Type: application/json` | send the documented content type |
| `413` | complete JSON request body exceeded 64 KiB | split the client request |
| `422` | JSON shape/type error or unsupported language direction | use the endpoint's documented fields and `en -> zh` |
| `503` plus `Retry-After: 1` on `/translate` or `/imme` | admission queue full or shutting down | retry with jitter or lower concurrency |
| `504` | end-to-end request timeout | inspect text length, concurrency, and host load |
| `500` | backend inference failure | inspect logs and `/readyz`; a Metal command failure poisons readiness |

CPU and Metal both segment long paragraphs through the shared tokenizer-aware
policy and preserve separator order. If a long request times out, inspect total
input size and host load; per-segment shape and work limits still apply.

Startup stops on invalid model files, checksum or architecture mismatches,
SentencePiece errors, Q8 validation errors, an unavailable Metal device, or a
Metal compilation/pipeline error.

Keep host publication on loopback unless a separate authenticated TLS proxy is
in front of it. The application itself is not an internet-facing gateway.
