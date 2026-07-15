# Operations guide

## Native macOS service

The installer creates the per-user LaunchAgent
`io.github.malusama.marian-mlx`, binds it to `127.0.0.1:3000` by default, and
stores data under `~/.local/share/marian-mlx`. A custom
`MARIAN_MLX_PORT=3100` is persisted, so operational checks must read the saved
port instead of assuming 3000:

```sh
MARIAN_MLX_HOME=${MARIAN_MLX_HOME:-"$HOME/.local/share/marian-mlx"}
if [ -r "$MARIAN_MLX_HOME/config/port" ]; then
  PORT=$(sed -n '1p' "$MARIAN_MLX_HOME/config/port")
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

### Installed-service lifecycle

`marian-mlxctl` is the supported operator interface; a source checkout is not
required:

```sh
~/.local/bin/marian-mlxctl status
~/.local/bin/marian-mlxctl verify
~/.local/bin/marian-mlxctl logs
~/.local/bin/marian-mlxctl restart
~/.local/bin/marian-mlxctl stop
~/.local/bin/marian-mlxctl start
~/.local/bin/marian-mlxctl update
~/.local/bin/marian-mlxctl rollback
~/.local/bin/marian-mlxctl uninstall
~/.local/bin/marian-mlxctl uninstall --purge
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
export MARIAN_MLX_HOME=/absolute/path/to/marian-mlx
export MARIAN_MLX_STATE=/absolute/path/to/marian-mlx-state
export MARIAN_MLX_BIN_DIR=/absolute/path/to/bin
"$MARIAN_MLX_BIN_DIR/marian-mlxctl" verify
```

The raw `launchctl` equivalents are useful for diagnosis, not normal lifecycle
management:

```sh
launchctl print "gui/$(id -u)/io.github.malusama.marian-mlx"
MARIAN_MLX_STATE=${MARIAN_MLX_STATE:-"$HOME/.local/state/marian-mlx"}
tail -f "$MARIAN_MLX_STATE/server.log" \
  "$MARIAN_MLX_STATE/server.error.log"
```

Repository scripts such as `scripts/install-macos.sh` are contributor entry
points when working from a source checkout. Installed users should use
`marian-mlxctl`.

### Production profile and controlled A/B

The v0.6 M1 qualification covers maximum batch 16, a 750 us window, about 32
concurrent short requests, FP32, and Flash `auto`. The resolved M1 profile is
`width=9`, `decode-rows=54`, `decode-steps=6`, `select-threads=256`, and
`custom-gemm-max=0`; `/info` reports the active values. A historical v0.4
mixed-f16 sweep found no benefit at concurrency 64, but that point is not a
current v0.6 FP32 qualification result.

The installed LaunchAgent does not inherit environment variables exported in
an interactive shell. The Metal variables below are for a separate foreground
source-build or release-binary A/B process. They do not reconfigure the
installed service:

| Environment variable | M1 production value | Valid values / purpose |
|---|---:|---|
| `MARIAN_MLX_METAL_PRECISION` | `fp32` | `fp32` or explicit `mixed-f16` storage |
| `MARIAN_MLX_METAL_PROFILE` | `auto` -> `m1` | `auto`, `m1`, `m2`, `m3`, `m4`, `generic` |
| `MARIAN_MLX_METAL_ATTENTION` | `auto` | `auto`, `classic`, `flash` |
| `MARIAN_MLX_METAL_FLASH_THRESHOLD` | `1` | positive self-attention sequence threshold, maximum 4096 |
| `MARIAN_MLX_METAL_FLASH_QUERY_TILE` | `4` | `1`, `2`, or `4` |
| `MARIAN_MLX_METAL_DUPLICATE_BATCH_WIDTH` | `9` | maximum physical occupancy width inside one dynamic batch |
| `MARIAN_MLX_METAL_DECODE_ROW_BUDGET` | `54` | positive rows multiplied by steps per submission |
| `MARIAN_MLX_METAL_DECODE_MAX_STEPS` | `6` | `1` through `8` |
| `MARIAN_MLX_METAL_DECODE_SELECTION_THREADS` | `256` | `128`, `256`, or `512` |
| `MARIAN_MLX_METAL_CUSTOM_GEMM_MAX_ROWS` | `0` | `0` disables custom FP32 GEMM; positive values set its row ceiling |

The product-level `MARIAN_MLX_METAL_*` names above are canonical.
`MARIAN_EDGE_METAL_*` spellings are accepted aliases for embedding use; if both
forms are set to different values, startup fails rather than choosing one.

For example, leave the LaunchAgent on its saved port and run the foreground
candidate on 3101:

```sh
MARIAN_MLX_METAL_ATTENTION=classic \
  target/release/marian-mlx-server \
  --backend metal --model-dir models/enzh --bind 127.0.0.1:3101
# In another shell:
curl -fsS http://127.0.0.1:3101/info
```

Duplicate width is not a result cache: core coalesces logical duplicates and
the Metal backend may rematerialize bounded physical rows for occupancy.
Re-sweep every device knob on a different Apple GPU instead of assuming the M1
knee applies unchanged. M2-M4 defaults are conservative, not qualified
performance claims.

## Runtime configuration

These variables map to server CLI flags for foreground and container runs.
The native installer persists only its documented install settings such as
port and CORS; shell exports do not alter an already installed LaunchAgent.

| Environment variable | Native default | Meaning |
|---|---|---|
| `MARIAN_MLX_BIND` | `127.0.0.1:3000` | complete listener address; container image overrides it to `0.0.0.0:3000` |
| `MARIAN_MLX_BACKEND` | `auto` | `auto`, `metal`, `cpu`, or development-only `echo` |
| `MARIAN_MLX_MODEL_DIR` | `models/enzh` | model directory; container image uses `/models/en-zh` |
| `MARIAN_MLX_CPU_THREADS` | `1` | CPU inference threads: `1`, `2`, or `4` |
| `MARIAN_MLX_QUEUE_CAPACITY` | `256` | bounded admission capacity |
| `MARIAN_MLX_MAX_BATCH_SIZE` | `16` | maximum logical dynamic batch size |
| `MARIAN_MLX_MAX_PADDED_SOURCE_CHARS` | `4096` | padded-character work bound for scheduler compatibility |
| `MARIAN_MLX_BATCH_WINDOW_US` | `750` | micro-batch collection window in microseconds |
| `MARIAN_MLX_REQUEST_TIMEOUT_MS` | `30000` | end-to-end scheduler timeout in milliseconds |
| `MARIAN_MLX_CORS_ORIGIN` | unset | one exact origin or `*`; keep unset unless the client requires CORS |
| `MARIAN_MLX_JSON_LOGS` | `false` | emit JSON rather than text tracing logs |

## Docker CPU service

The container listens on `0.0.0.0:3000` internally. The supported Compose file
publishes it only to host loopback; `MARIAN_MLX_HOST_PORT` changes the host side
without changing the container port:

```sh
docker compose pull
docker compose up -d
docker compose ps
docker compose logs -f
curl -fsS http://127.0.0.1:3000/info
docker compose down
```

Custom host port:

```sh
MARIAN_MLX_HOST_PORT=3100 docker compose up -d
curl -fsS http://127.0.0.1:3100/readyz
curl -fsS http://127.0.0.1:3100/info
# Immersive Translate: http://127.0.0.1:3100/imme
```

For a reproducible rollback, pin the immutable versioned image. Repeat the same
variables on later Compose commands because shell assignments are not stored:

```sh
MARIAN_MLX_IMAGE=ghcr.io/malusama/marian-mlx:cpu-0.6.0 \
MARIAN_MLX_HOST_PORT=3100 \
  docker compose pull
MARIAN_MLX_IMAGE=ghcr.io/malusama/marian-mlx:cpu-0.6.0 \
MARIAN_MLX_HOST_PORT=3100 \
  docker compose up -d
```

The named volume keeps the operator-downloaded model across container updates
and `docker compose down`. Use `docker compose down -v` only when intentionally
removing the model volume too. The image is Linux CPU-only on both Intel and
ARM hosts. Docker Desktop cannot pass the macOS Metal device into a Linux
container; use the native installer for Apple GPU inference.

### CPU ownership and compute threads

The CPU model has one owner. Concurrent HTTP requests still benefit from the
scheduler's micro-batching, and changing `MARIAN_MLX_CPU_THREADS` does not
create extra model replicas or workers. The setting accepts 1, 2, or 4 and is
applied at startup to FP32 matrix multiplication and Q8 rten/exact-AVX2 row
parallelism.

Raise the value one step at a time, recreate the container, warm the model with
representative traffic, and compare throughput, tail latency, CPU utilization,
and peak RSS:

```sh
MARIAN_MLX_CPU_THREADS=2 docker compose up -d --force-recreate
docker stats
```

Exact scaling depends on text length, batch shape, architecture, precision,
and model workspace; do not apply sizing numbers from the retired CPU runtime.

## Health and error contract

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

Startup fails closed for missing or invalid model artifacts, checksum or
precision/architecture mismatch, SentencePiece vocabulary errors, Q8 tensor
validation errors, an unavailable Metal device, MSL compilation failure, or
compute-pipeline creation failure.

Keep host publication on loopback unless a separate authenticated TLS proxy is
in front of it. The application itself is not an internet-facing gateway.
