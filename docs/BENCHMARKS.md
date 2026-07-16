# Benchmark notes

Results in different sections used different commits, host load, and workloads.
Compare rows within the same table; do not treat the largest number in this
file as a current product claim.

## Exploratory beam-width reference (2026-07-16)

The current Rust backends implement only `beam=1`, so this comparison used the
Mozilla Marian fork at commit
[`f31423c7`](https://github.com/mozilla/translations/tree/f31423c7c2c6ed8ae57d71a3d19a9db6f156060e)
with the same Q8 model artifact
(`4e5accc141373565ddc8fa1565bceaa8d0c3482a82cab8131c719ebcc6c2157c`).
It ran on the M1 test host with one CPU thread and mini-batch 16. The timed input
was the 1,012-sentence FLORES-200 `devtest` English source. Machine-readable
metadata and output hashes are in
[`beam-width-reference-m1pro.json`](../benchmarks/results/beam-width-reference-m1pro.json).

| Beam | Elapsed | Change from beam 1 | BLEU | chrF2++ |
|---:|---:|---:|---:|---:|
| 1 | 12.32 s | baseline | 38.8 | 25.0 |
| 2 | 16.45 s | +33.5% | 39.5 | 25.4 |
| 4 | 23.19 s | +88.2% | 39.7 | 25.6 |

Scores use SacreBLEU 2.5.1 with the Chinese BLEU tokenizer and chrF2++
(`char_order=6`, `word_order=2`). Paired bootstrap resampling with 1,000 samples
found both wider-beam outputs different from beam 1 at `p<=0.002` on both
metrics. Beam 4 gained only 0.2 BLEU and 0.2 chrF2++ over beam 2 while taking
41% longer than beam 2.

All five release golden sentences were unchanged. On the repository's
200-item engineering corpus, beam 1 to 2 changed 56 outputs, and beam 2 to 4
changed another 36. Spot checks found both improvements and regressions, so an
automatic-score gain is not a substitute for a webpage-focused human sample.

Mozilla's browser evaluation and quantization configurations also use
[`beam-size: 1`](https://github.com/mozilla/translations/blob/f31423c7c2c6ed8ae57d71a3d19a9db6f156060e/pipeline/eval/translators.py#L410-L425),
and its training guide uses beam 1 for tiny/base-memory students while using
beam 4 for the teacher. The current low-latency default therefore stays at 1.
Beam 2 is the measured speed/automatic-quality compromise for a future quality
mode; beam 4 is not a good default for this local-service workload.

## v0.6.0 pre-release candidate versus v0.1.0 (2026-07-16)

The v0.1.0 tag was rebuilt against its original MLX bridge and run immediately
before the final direct-Metal release candidate on the same Apple M1 / 16 GB
host, model, macOS 26.6 installation, HTTP driver, and batching settings. Each
entry is the median-by-throughput run from three measurements. The desktop was
under visible Otty and WindowServer load, so these paired live results are the
comparison boundary; the quieter historical peaks below remain useful but are
not mixed into this table.

| Runtime | Workload | Throughput | p50 | p95 | Peak RSS | Output hashes/run |
|---|---|---:|---:|---:|---:|---:|
| v0.1.0 MLX FP32 | 1,000 short requests | 486.64 item/s | 66.74 ms | 77.80 ms | 205,632 KiB | 1 |
| v0.6.0 pre-release candidate, direct Metal FP32 | 1,000 short requests | 546.19 item/s | 63.09 ms | 68.20 ms | 252,240 KiB | 1 |
| v0.1.0 MLX FP32 | 5 x 200-item corpus | 116.68 item/s | 1622.82 ms | 2116.16 ms | 214,448 KiB | 5 |
| v0.6.0 pre-release candidate, direct Metal FP32 | 5 x 200-item corpus | 149.14 item/s | 1336.02 ms | 1385.14 ms | 257,088 KiB | 1 |

The candidate is 12.2% faster than the first release on repeated short
traffic and 27.8% faster on 200 distinct items. It is also 10.1% and 18.0%
faster than a live v0.5.0 binary measured in the same loaded desktop window.
The v0.1.0 corpus produced a different response hash for every one of its five
identical request bodies; the final runtime produced one stable hash. Peak RSS
is recorded for deployment sizing but includes macOS shared-GPU and purgeable
page behavior, so it is not used as an allocator comparison.

### Candidate FlashAttention-style A/B

This A/B changes only `MARIAN_EDGE_METAL_ATTENTION` in the candidate binary. Flash
and classic produced identical output hashes for both workloads.

| Workload | Classic | Flash q4 auto | Throughput change | Flash p50 | Flash p95 |
|---|---:|---:|---:|---:|---:|
| 1,000 short requests | 486.50 item/s | 546.19 item/s | +12.3% | 63.09 ms | 68.20 ms |
| 5 x 200-item corpus | 142.16 item/s | 149.14 item/s | +4.9% | 1336.02 ms | 1385.14 ms |

The kernel streams 32-key tiles, handles query tiles 1/2/4 with online softmax,
accepts packed QKV/KV strides and offsets, and never materializes the classic
O(N^2) score matrix. Its low-level oracle covers key lengths 31/32/33 and
multiple heads/dimensions; real Metal FP32 matched CPU FP32 on 200/200 corpus
items. Mixed-F16 remained explicit at 198/200, with mismatches 116 and 192.

### Candidate Metal trace

A 300-request Instruments trace recorded 40 submitted and 40 completed command
buffers, zero errors, and no command buffer without GPU intervals. Labels show
20 fused `encode+cross-cache` submissions and 20 six-token decode submissions.
GPU active time was 510.32 ms; per-command-buffer GPU p50/p95 were 10.46/27.96
ms. The compact checked-in evidence is
[`benchmarks/results/v0.6.0-m1.json`](../benchmarks/results/v0.6.0-m1.json);
`tools/profile_metal.sh` regenerates the trace, exported evidence tables, and
summary JSON.

The candidate was measured before the package version changed from 0.5.0 to
0.6.0; the qualification artifact records that state and the measured binary
hash explicitly. From the v0.6.0 release commit, reproduce its source-diff hash
with this canonical command. The `crates` scope intentionally binds runtime,
kernel, manifest, and test changes while excluding later documentation-only
edits:

```sh
git diff --binary 3a6c2d9240eaaa2a56135a61b7e7d721de061e36 \
  v0.6.0 -- crates \
  | shasum -a 256
```

## Historical v0.5.0 qualification (2026-07-15)

The optimized implementation is commit `6c056a6648b5c2581747e89a4aac594094d9b1d8`
on an Apple M1 / 16 GB host running macOS 26.6. It uses the same Mozilla en-zh
model and server settings as the formal v0.1.0 macOS release: maximum batch 16,
750 us batch window, FP32 storage, and concurrency 32 for the short request.
The v0.1.0 archive SHA-256 was
`3d6a343981ec8e88d4ef1857a09ad57ff324f967c13cf32ff3515cf42f2ce4f1`;
its server reported revision `9d7063fe0c4d` and MLX FP32. Throughput and
latency entries are three-run medians after per-run warmup. Optimized peak RSS
is the maximum sampled during a fresh-process run because macOS may later
reclaim purgeable model pages from a long-lived process.

| Runtime | Workload | Throughput | p50 | p95 | Peak RSS | Repeated output hashes |
|---|---|---:|---:|---:|---:|---:|
| v0.1.0 MLX FP32 | 1,000 short requests | 536.57 item/s | 62.64 ms | 75.44 ms | 231,008 KiB | 1 |
| v0.5.0 direct Metal FP32 + Flash q4 | 1,000 short requests | 599.32 item/s | 53.11 ms | 65.73 ms | 242,064 KiB | 1 |
| v0.1.0 MLX FP32 | 10 x 200-item corpus | 122.09 item/s | 1617.93 ms | 1764.13 ms | 233,872 KiB | 10 per run |
| v0.5.0 direct Metal FP32 + Flash q4 | 10 x 200-item corpus | 165.29 item/s | 1200.01 ms | 1290.99 ms | 217,120 KiB | 1 |

The v0.5.0 direct-Metal runtime is 11.7% faster than the formal v0.1.0 release
on the repeated short request and 35.4% faster on the corpus. Median p50 fell
15.2% and 25.8% respectively. It also remains free of the MLX/C++ runtime
dependency and deterministic in the repeated corpus test. Every v0.1.0 corpus
run produced ten distinct output hashes from ten identical sequential request
bodies; v0.5.0 produced one. That observation is reproducible behavior, not a
translation-quality judgment.

The short workload intentionally remains the same repeated sentence used to
qualify v0.1.0. v0.5.0 coalesces byte-for-byte duplicates only inside the
current dynamic batch, retains nine rows on M1 to avoid the MPS small-matrix
efficiency cliff, and never serves a cached result across batches. The corpus
contains 200 distinct items; its gain comes from fuller source-length buckets,
GPU-resident decode chunks, MPS GEMM, and persistent buffer arenas rather than
duplicate coalescing.

## Flash q4 versus classic direct Metal

This controlled A/B keeps the current Rust/Metal host, FP32 weights, request
driver, model, and batching fixed, changing only
`MARIAN_EDGE_METAL_ATTENTION`. It uses 1,000 short requests and ten corpus
requests after warmup.

| Workload | Attention | Throughput | p50 | p95 |
|---|---|---:|---:|---:|
| short request | classic | 306.28 item/s | 101.17 ms | 130.81 ms |
| short request | Flash q4 | 314.98 item/s | 100.71 ms | 118.92 ms |
| 200-item corpus | classic | 79.67 item/s | 2531.45 ms | 2623.80 ms |
| 200-item corpus | Flash q4 | 82.46 item/s | 2414.51 ms | 2556.67 ms |

Flash q4 improved throughput by 2.8% and 3.5% respectively. It fuses score,
masking, online softmax, and value accumulation and does not allocate the
classic path's O(N^2) score buffer. A padded low-level numerical oracle passes
within `2e-5`; the real FP32 Metal backend matched CPU FP32 on all 200 corpus
items and all release golden/long-text/output-budget cases.

The encoder-isolated test repeats one unbroken phrase and limits decoding to
one output token. It shows the benefit once attention becomes a larger share
of the request:

| Repeated phrases | Classic p50 | Flash q4 p50 | Change |
|---:|---:|---:|---:|
| 1 | 17.17 ms | 16.54 ms | -3.7% |
| 10 | 33.46 ms | 32.79 ms | -2.0% |
| 20 | 40.92 ms | 40.13 ms | -1.9% |
| 40 | 66.20 ms | 50.48 ms | -23.8% |
| 80 | 118.12 ms | 103.69 ms | -12.2% |
| 160 | 285.12 ms | 236.00 ms | -17.2% |
| 320 | 784.58 ms | 576.29 ms | -26.5% |

## M1 settings tested for v0.6.0

The qualified exact-output configuration is FP32, Flash `auto`, maximum batch
16, a 750 us batching window, and about 32 concurrent short requests. `/info`
reports the complete resolved profile:

```text
m1(width=9,decode-rows=54,decode-steps=6,select-threads=256,custom-gemm-max=0)
```

The defaults are the deployment setting. Environment overrides exist for
controlled local sweeps, not as required production configuration:

```sh
target/release/marian-edge-server --backend metal --model-dir models/enzh \
  --max-batch-size 16 --batch-window-us 750
```

| M1 knob | Default | Qualification result |
|---|---:|---|
| Flash query tile | 4 | Tiles 1/2/4 pass the classic oracle; q4 wins the production shapes. |
| Duplicate physical width | 9 | Width 7/8/9 reached 499.53/563.18/592.11 item/s in the focused sweep. |
| Decode row budget | 54 | Fills up to six steps for nine retained rows without crossing the measured occupancy knee. |
| Decode maximum steps | 6 | Improves submission amortization; the next submission contracts to one step after newly observed completion. |
| Selection threads | 256 | Faster than 128 in the corpus sweep; 512 regressed on M1. |
| Custom FP32 GEMM rows | 0 | The row-9/16 custom microtile paths were slower than shape-cached MPS and remain disabled. |

The 1,000-request three-run median above is the release comparison, not a
shorter tuning sweep. M2-M4 profiles remain conservative until measured on the
corresponding hardware.

### Prior v0.4 precision and concurrency sweep

At maximum batch 16 and a 750 us window, the mixed-f16 concurrency sweep used
600 short requests per point:

| Concurrency | Throughput | p50 | p95 | Peak RSS |
|---:|---:|---:|---:|---:|
| 1 | 30.53 item/s | 34.07 ms | 41.90 ms | 160,880 KiB |
| 8 | 117.12 item/s | 71.76 ms | 82.91 ms | 162,048 KiB |
| 16 | 192.62 item/s | 86.31 ms | 99.55 ms | 162,448 KiB |
| 32 | 322.48 item/s | 97.57 ms | 117.23 ms | 162,608 KiB |
| 64 | 324.40 item/s | 193.27 ms | 232.46 ms | 164,096 KiB |

Concurrency 32 is the throughput/latency knee. At concurrency 32, a separate
8/16/32 maximum-batch and 0/250/750/1500 us window sweep selected batch 16 and
750 us at 309.40 item/s. Batch 32 did not improve throughput, while a zero
window effectively disabled useful coalescing and fell to about 34 item/s.

Three-run medians at the selected settings show the precision trade-off:

| Precision | Workload | Throughput | p50 | p95 | Peak RSS | FP32 corpus parity |
|---|---|---:|---:|---:|---:|---:|
| FP32 | short request | 326.02 item/s | 98.24 ms | 120.25 ms | 242,688 KiB | exact contract |
| mixed-f16 | short request | 324.78 item/s | 94.74 ms | 119.24 ms | 181,584 KiB | 198/200 |
| FP32 | 200-item corpus | 80.42 item/s | 2482.21 ms | 2632.25 ms | 246,384 KiB | exact contract |
| mixed-f16 | 200-item corpus | 80.92 item/s | 2458.92 ms | 2672.58 ms | 162,832 KiB | 198/200 |

The v0.4 mixed-F16 measurements are retained as historical precision evidence;
mixed-F16 was not requalified for the v0.5 performance table. Use it only when
saving memory is worth the known 2/200 output delta. For latency-oriented
traffic, concurrency 8-16 remains the better operating range. The CPU fallback
remains one compute thread on this M1 because the fixed 384-dimensional graph
does not amortize Rayon overhead.

## Previous direct Metal benchmark (2026-07-15)

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

Reproduce the HTTP measurements against the backend that is actually running:

```sh
SERVICE_ORIGIN=${SERVICE_ORIGIN:-http://127.0.0.1:3000}
PORT=${SERVICE_ORIGIN##*:}
python3 tools/bench_http.py \
  --url "$SERVICE_ORIGIN/translate" \
  --requests 500 --concurrency 32 --warmup 32 \
  --model-dir models/enzh --pid "$(lsof -tiTCP:"$PORT" -sTCP:LISTEN)" \
  --threads 1 --commit "$(git rev-parse HEAD)" --output result.json

python3 tools/bench_http.py \
  --url "$SERVICE_ORIGIN/imme" \
  --corpus benchmarks/corpus-v1.jsonl \
  --requests 5 --concurrency 1 --warmup 1 \
  --model-dir models/enzh --pid "$(lsof -tiTCP:"$PORT" -sTCP:LISTEN)" \
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

## Historical Q8 CPU measurements

This section records historical engineering measurements. The exact five-item
golden expectations remain in tests, but the raw 200-item per-output comparison
and allocation-run logs are not checked-in artifacts. Treat the figures below
as provenance-bound evidence, not as a currently reproducible release gate.

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
