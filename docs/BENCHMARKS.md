# Benchmark status

The direct Metal backend does not yet have a published benchmark result. Do
not relabel the numbers below: they were measured with the retired MLX v0.32.0
implementation and do not establish the throughput or latency of the Rust
`objc2-metal` host and custom MSL kernels.

Before publishing a direct Metal comparison, run the same converted FP32
weights and request corpus, verify translation parity against the trusted
reference, report warmup and batch settings, capture peak memory, and attach an
Instruments Metal System Trace. Report the exact commit, hardware, macOS
version, concurrency, and sample count. The release is a single executable;
there is no `libmlx.dylib` or external `.metallib` to identify in new runs.

## Retired MLX/Bergamot baseline (2026-07-14)

This section is retained only as a migration baseline. Neither row describes a
current runtime: both MLX and the Bergamot/C++ worker have been removed. It was
measured on an Apple M1 / 16 GB Mac running macOS 26.6. The native `arm64`
process used MLX v0.32.0, FP32 weights, a lexical shortlist,
`max_batch_size=16`, and a 750 us batch window. The retired CPU run used the
ARM64 Docker image, one Bergamot worker, and `int8shiftAlphaAll`.

Request text: `The weather is beautiful today.` Each measurement used 32
shape-warmup requests followed by 500 measured requests at concurrency 32.

| Historical runtime | Throughput | Mean | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|
| Retired Bergamot int8 Docker, 1 worker | 95.61 req/s | 323.87 ms | 301.10 ms | 504.94 ms | 1245.97 ms |
| Retired MLX/Metal FP32 | 536.04 req/s | 57.23 ms | 58.72 ms | 75.86 ms | 87.03 ms |

On that historical workload, the MLX service delivered 5.61x the throughput of
the one-worker Bergamot run and reduced p95 by 85%. A separate 4,000-request
sustained MLX run completed at 529.50 req/s with p95 78.22 ms.

Historical MLX concurrency sweep (300 measured requests at each level):

| Concurrency | Throughput | p50 | p95 | p99 |
|---:|---:|---:|---:|---:|
| 1 | 41.87 req/s | 22.41 ms | 31.02 ms | 31.89 ms |
| 8 | 195.26 req/s | 34.30 ms | 63.48 ms | 66.93 ms |
| 16 | 302.30 req/s | 54.63 ms | 65.93 ms | 67.06 ms |
| 32 | 648.15 req/s | 45.02 ms | 60.23 ms | 63.07 ms |
| 64 | 514.65 req/s | 111.55 ms | 130.56 ms | 142.72 ms |

The old peak varied between runs. Its concurrency recommendation must not be
carried over to the direct Metal backend without a new sweep.

## Current Q8 qualification status

The pure-Rust Q8 graph passes all five release golden translations exactly. In
a 200-item differential corpus against the retired CPU reference, 164 outputs
were exact matches. The remaining items include near-tie token choices, so the
result is a compatibility measurement, not a claim of bit-for-bit equivalence
or a general translation-quality score. Repeated 80-sentence input and newline
preservation also matched the retired long-text baseline.

These checks are correctness gates, not a published throughput benchmark. A
current CPU benchmark still needs exact commit, artifact checksums, host,
thread count, warmup, concurrency, corpus, peak RSS, and latency percentiles.

### Historical GPU trace

An Instruments `Metal System Trace` was attached to the retired MLX process
during the sustained run. It captured 5,801 application command-buffer
submissions, 28,087 GPU execution-point rows, and 6,608 completed
command-buffer rows. The backend info response independently reported
`device: Apple M1`. These observations prove GPU use for that historical run,
not for the replacement backend.
