# Operations guide

## Native macOS service

The installer creates a per-user LaunchAgent named
`io.github.malusama.marian-mlx`. It binds to `127.0.0.1:3000` by default and
stores data under `~/.local/share/marian-mlx`.

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

Uninstall the service while preserving the downloaded model:

```sh
scripts/uninstall-macos.sh
```

Use `scripts/uninstall-macos.sh --purge` to remove releases, model, cache, and
logs too.

## Docker CPU service

```sh
docker compose up -d
docker compose ps
docker compose logs -f
curl -fsS http://127.0.0.1:3000/info
docker compose down
```

The named volume keeps the operator-downloaded model across container updates.
The image is Linux CPU-only on both Intel and ARM hosts. Docker Desktop cannot
pass the macOS Metal device into a Linux container; use the native installer
for Apple GPU inference.

### CPU worker sizing

Compose defaults `MARIAN_MLX_CPU_THREADS` to `1`. A single worker still receives the
micro-batches formed from concurrent HTTP requests, and is the safest setting
for small ARM systems, NAS hosts, and Docker Desktop.

In an ARM64 smoke test with the released `base-memory` model, observed peak RSS
was about 0.4-0.5 GB with one active worker and 0.7-0.8 GB with two. Replicas
load lazily, so memory can rise only after concurrency reaches a newly enabled
worker. Exact usage depends on text length, batch shape, architecture, and the
configured model workspace.

To test two workers:

```sh
MARIAN_MLX_CPU_THREADS=2 docker compose up -d --force-recreate
docker stats
```

Give a one-worker container at least 768 MiB before adding a tight memory
limit; allow at least 1 GiB when trying two. Raise the value one step at a time,
warm every worker with representative traffic, and compare throughput, tail
latency, and peak RSS. Revert to one if memory pressure, swapping, or OOM kills
appear. Values above two are intended for measured high-throughput deployments,
not routine local use.

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
- startup fails closed: inspect the error log for a missing model, checksum,
  architecture, MLX library, or Metal library.

Keep the listener on loopback unless a separate authenticated TLS proxy is in
front of it. The application itself is not an internet-facing gateway.
