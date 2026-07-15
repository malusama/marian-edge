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
FP32, Flash `auto`, and the default duplicate width 9. The last value can be
overridden before service start with
`MARIAN_MLX_METAL_DUPLICATE_BATCH_WIDTH`; values must be positive. It controls
how many identical rows are retained inside one dynamic batch and is not a
result cache. Re-sweep it on a different Apple GPU instead of assuming the M1
small-matrix knee applies unchanged.

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
