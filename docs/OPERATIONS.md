# Operations guide

## Native macOS service

The installer creates a per-user LaunchAgent named
`io.github.malusama.marian-mlx`. It binds to `127.0.0.1:3000` by default and
stores data under `~/.local/share/marian-mlx`.

The native release payload is one server executable. Its MSL source is embedded
and compiled through the system Metal framework when the process starts; there
is no adjacent `libmlx.dylib` or `.metallib` to install or verify. The model
directory is still managed separately.

```sh
# Process and launchd state
launchctl print "gui/$(id -u)/io.github.malusama.marian-mlx"

# Readiness and backend details
curl -fsS http://127.0.0.1:3000/readyz
curl -fsS http://127.0.0.1:3000/info

# Logs
tail -f ~/.local/state/marian-mlx/server.log \
  ~/.local/state/marian-mlx/server.error.log

# Restart
launchctl kickstart -k "gui/$(id -u)/io.github.malusama.marian-mlx"
```

Re-run `scripts/install-macos.sh` to update. The installer downloads and
verifies the new runtime before stopping the current service, then checks
`/readyz`; a failed cutover is rolled back.

The qualified M1 throughput settings are maximum batch 16, a 750 us window,
about 32 concurrent short requests, FP32, and Flash `auto`. The resolved M1
profile is `width=9`, `decode-rows=54`, `decode-steps=6`,
`select-threads=256`, and `custom-gemm-max=0`; `/info` reports these values at
runtime. No environment overrides are required for production.

For controlled A/B work, every Metal setting is explicit:

| Environment variable | M1 production value | Valid values / purpose |
|---|---:|---|
| `MARIAN_MLX_METAL_PRECISION` | `fp32` | `fp32` or explicit `mixed-f16` storage |
| `MARIAN_MLX_METAL_PROFILE` | `auto` -> `m1` | `auto`, `m1`, `m2`, `m3`, `m4`, `generic` |
| `MARIAN_MLX_METAL_ATTENTION` | `auto` | `auto`, `classic`, `flash` |
| `MARIAN_MLX_METAL_FLASH_THRESHOLD` | `1` | Positive self-attention sequence threshold, maximum 4096 |
| `MARIAN_MLX_METAL_FLASH_QUERY_TILE` | `4` | `1`, `2`, or `4` |
| `MARIAN_MLX_METAL_DUPLICATE_BATCH_WIDTH` | `9` | Positive physical occupancy width inside one dynamic batch |
| `MARIAN_MLX_METAL_DECODE_ROW_BUDGET` | `54` | Positive rows multiplied by steps per submission |
| `MARIAN_MLX_METAL_DECODE_MAX_STEPS` | `6` | `1` through `8` |
| `MARIAN_MLX_METAL_DECODE_SELECTION_THREADS` | `256` | `128`, `256`, or `512` |
| `MARIAN_MLX_METAL_CUSTOM_GEMM_MAX_ROWS` | `0` | `0` disables custom FP32 GEMM; positive values set its row ceiling |

Duplicate width is not a result cache. Re-sweep every device knob on a
different Apple GPU instead of assuming the M1 knee applies unchanged. M2-M4
defaults are conservative, not qualified performance claims.

Uninstall the service while preserving the downloaded model:

```sh
scripts/uninstall-macos.sh
```

Use `scripts/uninstall-macos.sh --purge` to remove releases, model, cache, and
logs too.

## Docker CPU service

```sh
docker compose pull
docker compose up -d
docker compose ps
docker compose logs -f
curl -fsS http://127.0.0.1:3000/info
docker compose down
```

The named volume keeps the operator-downloaded model across container updates.
The image is Linux CPU-only on both Intel and ARM hosts and runs the pure-Rust
Q8 backend. Docker Desktop cannot pass the macOS Metal device into a Linux
container; use the native installer for Apple GPU inference.

### CPU ownership and compute threads

The CPU model has one owner. Concurrent HTTP requests still benefit from the
scheduler's micro-batching, and changing `MARIAN_MLX_CPU_THREADS` does not
create extra model replicas or workers. The setting accepts 1, 2, or 4 and is
applied at startup to both `MATMUL_NUM_THREADS` and `RAYON_NUM_THREADS`: it
controls FP32 matrix multiplication as well as Q8 rten/exact-AVX2 row
parallelism.

Raise the value one step at a time, recreate the container, warm the model with
representative traffic, and compare throughput, tail latency, CPU utilization,
and peak RSS:

```sh
MARIAN_MLX_CPU_THREADS=2 \
  docker compose up -d --force-recreate
docker stats
```

Exact scaling depends on text length, batch shape, architecture, precision,
and model workspace; do not apply sizing numbers from the retired CPU runtime.

## Health contract

| Endpoint | Success means | Suitable for |
|---|---|---|
| `/livez` | HTTP loop is alive | process liveness |
| `/readyz` | model worker accepts requests | traffic readiness |
| `/health` | compatibility endpoint is reachable | existing clients |
| `/info` | backend/device/model identity | deployment verification |
| `/metrics` | Prometheus text is available | capacity monitoring |

`/health` alone does not prove that a model is loaded. Gate traffic on
`/readyz` and inspect `/info` after every deployment.

## Capacity and failures

- `503` with `Retry-After: 1`: admission queue is full or shutting down;
  retry with jitter or lower concurrency.
- `504`: request exceeded its end-to-end timeout; check text length and host
  load.
- `422`: invalid/unsupported direction; the current release supports only
  English to Chinese.
- startup fails closed: inspect the error log for a missing model, checksum or
  precision/architecture mismatch, SentencePiece vocabulary error, Q8 tensor
  validation error, unavailable Metal device, MSL compilation failure, or
  compute-pipeline creation failure.

The CPU backend segments long paragraphs before inference and preserves their
separator order. If a long request times out, inspect total input size and host
load; the per-chunk shape and work limits still apply even after segmentation.

Keep the listener on loopback unless a separate authenticated TLS proxy is in
front of it. The application itself is not an internet-facing gateway.
