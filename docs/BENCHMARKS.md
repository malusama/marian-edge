# Current direct Metal benchmark (2026-07-15)

This is the first result for the Rust `objc2-metal` host and embedded MSL
kernels. It is not the retired MLX result below.

| Field | Value |
|---|---|
| commit | `6fdcf8d1f68167261487353e910c1f7b7fb31e02` |
| host | Apple M1, 16 GB, macOS 26.6 (`25G5052e`) |
| model | Mozilla en-zh `base-memory`, converted FP32 safetensors |
| weights SHA-256 | `fcd6f7a791293b6f9b6a959b7e9ee856a34d451afecaed2dcb5ac314b47f6967` |
| corpus SHA-256 | `41fbef085648815e15bd7ec7261816c9d55c1f04c2489cab3e7550887449de0b` |
| server batching | maximum 16 items, 750 us window |

The single-sentence workload used `The weather is beautiful today.`, 32 warmup
requests, then 500 requests at concurrency 32. The corpus workload sent the
checked-in 200-item corpus to `/imme`, one warmup request, then five measured
requests (1,000 translated items) at concurrency 1.

| Precision mode | Workload | Throughput | p50 | p95 | p99 | Peak RSS |
|---|---|---:|---:|---:|---:|---:|
| FP32, default | single sentence | 289.53 item/s | 115.60 ms | 135.05 ms | 142.17 ms | 241,824 KiB |
| mixed-f16, explicit | single sentence | 357.17 item/s | 88.15 ms | 105.72 ms | 108.42 ms | 180,992 KiB |
| FP32, default | 200-item corpus | 88.22 item/s | 2188.85 ms | 2530.86 ms | 2530.86 ms | 243,312 KiB |
| mixed-f16, explicit | 200-item corpus | 91.96 item/s | 2157.09 ms | 2339.85 ms | 2339.85 ms | 157,696 KiB |

On this run, explicit mixed-f16 storage improved single-sentence throughput by
23.4% and corpus throughput by 4.2%, while reducing measured peak RSS by 25.2%
and 35.2% respectively. FP32 remains the default output contract. In the
CPU-FP32 versus Metal differential test, FP32 was exact on all 200 items;
mixed-f16 was exact on 198/200, with differences at indexes 116 and 192. The
mixed mode therefore remains explicit and reports itself as `mixed-f16` from
`/info`.

An 11.13-second Instruments Metal System Trace attached to this server recorded
280 command-submission rows, 2,126 completed-table rows, 88,114 GPU
execution-point rows, and no command errors. Table rows are profiler evidence,
not a count of unique application command buffers. Reproduce the capture with
`tools/profile_metal.sh`; the 117 MB trace itself is not checked into Git.

Reproduce the HTTP measurements with:

```sh
python3 tools/bench_http.py \
  --url http://127.0.0.1:3000/translate \
  --requests 500 --concurrency 32 --warmup 32 \
  --model-dir models/enzh --pid "$(lsof -tiTCP:3000 -sTCP:LISTEN)" \
  --threads 1 --commit "$(git rev-parse HEAD)" --output result.json

python3 tools/bench_http.py \
  --url http://127.0.0.1:3000/imme \
  --corpus benchmarks/corpus-v1.jsonl \
  --requests 5 --concurrency 1 --warmup 1 \
  --model-dir models/enzh --pid "$(lsof -tiTCP:3000 -sTCP:LISTEN)" \
  --threads 1 --commit "$(git rev-parse HEAD)" --output corpus-result.json
```

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

## Current Q8 CPU qualification and allocation result

The pure-Rust Q8 graph passes all five release golden translations exactly. In
a 200-item differential corpus against the retired CPU reference, 164 outputs
were exact matches. The remaining items include near-tie token choices, so the
result is a compatibility measurement, not a claim of bit-for-bit equivalence
or a general translation-quality score. Repeated 80-sentence input and newline
preservation also matched the retired long-text baseline.

The allocation/data-flow implementation was compared on the same M1, one CPU
thread, with Q8 artifact SHA-256
`4e5accc141373565ddc8fa1565bceaa8d0c3482a82cab8131c719ebcc6c2157c`.
The baseline was `547f66f54c2a`; the optimized runtime code is in
`6fdcf8d1f681`. Warm single-sentence and 200-item corpus measurements stayed
within sub-one-percent run variance while reducing peak RSS.

| Workload | Metric | Baseline | Optimized | Change |
|---|---|---:|---:|---:|
| warm single sentence | p50 | 6.909 ms | 6.981 ms | +1.0% |
| warm single sentence | throughput | 141.44 item/s | 141.06 item/s | -0.3% |
| warm single sentence | peak RSS | 202,784 KiB | 159,664 KiB | -21.3% |
| 200-item corpus | p50 | 981.82 ms | 977.88 ms | -0.4% |
| 200-item corpus | throughput | 203.53 item/s | 203.26 item/s | -0.1% |
| 200-item corpus | peak RSS | 143,376 KiB | 136,624 KiB | -4.7% |

The real-artifact golden run passed 2/2, including the five release outputs,
long text, newline preservation, and segmented output budget. Its load report
was 18,874,368 canonical-weight bytes, 19,206,144 packed-weight bytes,
24,576,000 embedding bytes, and 13.82 ms to build packed weights. Canonical
Q8 rows are retained because exact GEMV and lexical-shortlist access use them.

### Historical GPU trace

An Instruments `Metal System Trace` was attached to the retired MLX process
during the sustained run. It captured 5,801 application command-buffer
submissions, 28,087 GPU execution-point rows, and 6,608 completed
command-buffer rows. The backend info response independently reported
`device: Apple M1`. These observations prove GPU use for that historical run,
not for the replacement backend.
