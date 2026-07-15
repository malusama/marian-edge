#!/bin/sh
set -eu

ROOT=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
URL=${1:-http://127.0.0.1:3000/translate}
OUTPUT=${2:-"$ROOT/benchmarks/metal-system.trace"}
DURATION=${MARIAN_METAL_TRACE_SECONDS:-10s}
REQUESTS=${MARIAN_METAL_TRACE_REQUESTS:-300}
CONCURRENCY=${MARIAN_METAL_TRACE_CONCURRENCY:-32}

PORT=$(printf '%s' "$URL" | sed -E 's|^[a-z]+://[^:/]+:([0-9]+).*|\1|')
PID=$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t | head -n 1)
[ -n "$PID" ] || {
  echo "no listening process found for $URL" >&2
  exit 1
}

rm -rf "$OUTPUT" "$OUTPUT.toc.xml"
xcrun xctrace record \
  --template 'Metal System Trace' \
  --attach "$PID" \
  --time-limit "$DURATION" \
  --output "$OUTPUT" &
TRACE_PID=$!
sleep 1
python3 "$ROOT/tools/bench_http.py" \
  --url "$URL" \
  --requests "$REQUESTS" \
  --concurrency "$CONCURRENCY" \
  --warmup 0 \
  --pid "$PID" \
  --output "$OUTPUT.benchmark.json" >/dev/null
wait "$TRACE_PID"
xcrun xctrace export --input "$OUTPUT" --toc --output "$OUTPUT.toc.xml"
echo "trace: $OUTPUT"
echo "benchmark: $OUTPUT.benchmark.json"
echo "toc: $OUTPUT.toc.xml"
