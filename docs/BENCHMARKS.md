# M1 benchmark

Measured 2026-07-14 on an Apple M1 / 16 GB Mac running macOS 26.6. The MLX
runtime was a native `arm64` release build with MLX v0.32.0, FP32 weights,
lexical shortlist, `max_batch_size=16`, and a 750 us batch window. The CPU run
used the ARM64 Docker image, one Bergamot worker, and `int8shiftAlphaAll`.

Request text: `The weather is beautiful today.` Each measurement used 32
shape-warmup requests followed by 500 measured requests at concurrency 32.

| Runtime | Throughput | Mean | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|
| Bergamot int8 Docker, 1 worker | 95.61 req/s | 323.87 ms | 301.10 ms | 504.94 ms | 1245.97 ms |
| MLX/Metal FP32 | 536.04 req/s | 57.23 ms | 58.72 ms | 75.86 ms | 87.03 ms |

On this workload the MLX service delivered 5.61x throughput and reduced p95 by
85%. A separate 4,000-request sustained MLX run completed at 529.50 req/s with
p95 78.22 ms.

MLX concurrency sweep (300 measured requests at each level):

| Concurrency | Throughput | p50 | p95 | p99 |
|---:|---:|---:|---:|---:|
| 1 | 41.87 req/s | 22.41 ms | 31.02 ms | 31.89 ms |
| 8 | 195.26 req/s | 34.30 ms | 63.48 ms | 66.93 ms |
| 16 | 302.30 req/s | 54.63 ms | 65.93 ms | 67.06 ms |
| 32 | 648.15 req/s | 45.02 ms | 60.23 ms | 63.07 ms |
| 64 | 514.65 req/s | 111.55 ms | 130.56 ms | 142.72 ms |

The exact peak varies between runs; concurrency 32 is the best starting point
for this M1 and short sentence. Real traffic needs its own length distribution
and batch-window sweep.

## GPU trace

An Instruments `Metal System Trace` was attached to the release process during
the sustained run. It captured 5,801 application command-buffer submissions,
28,087 GPU execution-point rows, and 6,608 completed command-buffer rows. The
backend info response independently reported `device: Apple M1`.
