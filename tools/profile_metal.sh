#!/bin/sh
set -eu

ROOT=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
URL=${1:-http://127.0.0.1:3000/translate}
OUTPUT=${2:-"$ROOT/benchmarks/metal-system.trace"}
DURATION=${MARIAN_METAL_TRACE_SECONDS:-10s}
REQUESTS=${MARIAN_METAL_TRACE_REQUESTS:-300}
CONCURRENCY=${MARIAN_METAL_TRACE_CONCURRENCY:-32}
READY_TIMEOUT=${MARIAN_METAL_TRACE_READY_TIMEOUT_SECONDS:-30}
EVIDENCE="$OUTPUT.evidence"
SUMMARY="$OUTPUT.summary.json"
TOC="$OUTPUT.toc.xml"

fail() {
  echo "profile_metal: $*" >&2
  exit 1
}

for command in lsof notifyutil python3 xcrun; do
  command -v "$command" >/dev/null 2>&1 || fail "required command not found: $command"
done

case "$REQUESTS" in
  ''|*[!0-9]*) fail "MARIAN_METAL_TRACE_REQUESTS must be a positive integer" ;;
esac
case "$CONCURRENCY" in
  ''|*[!0-9]*) fail "MARIAN_METAL_TRACE_CONCURRENCY must be a positive integer" ;;
esac
case "$READY_TIMEOUT" in
  ''|*[!0-9]*) fail "MARIAN_METAL_TRACE_READY_TIMEOUT_SECONDS must be a positive integer" ;;
esac
[ "$REQUESTS" -gt 0 ] || fail "MARIAN_METAL_TRACE_REQUESTS must be positive"
[ "$CONCURRENCY" -gt 0 ] || fail "MARIAN_METAL_TRACE_CONCURRENCY must be positive"
[ "$READY_TIMEOUT" -gt 0 ] || fail "MARIAN_METAL_TRACE_READY_TIMEOUT_SECONDS must be positive"

PORT=$(printf '%s' "$URL" | sed -E 's|^[a-z]+://[^:/]+:([0-9]+).*|\1|')
case "$PORT" in
  ''|*[!0-9]*) fail "URL must contain an explicit numeric port: $URL" ;;
esac
PID=$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | sed -n '1p')
[ -n "$PID" ] || fail "no listening process found for $URL"

READY_FILE=${TMPDIR:-/tmp}/marian-edge-metal-trace-ready.$$
NOTIFICATION=io.github.malusama.marian-edge.metal-trace-ready.$$
TRACE_PID=
READY_PID=

cleanup() {
  status=$?
  trap - EXIT HUP INT TERM
  if [ -n "$READY_PID" ] && kill -0 "$READY_PID" 2>/dev/null; then
    kill "$READY_PID" 2>/dev/null || true
  fi
  if [ "$status" -ne 0 ] && [ -n "$TRACE_PID" ] && kill -0 "$TRACE_PID" 2>/dev/null; then
    kill "$TRACE_PID" 2>/dev/null || true
  fi
  rm -f "$READY_FILE"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' HUP INT TERM

rm -rf "$OUTPUT" "$TOC" "$SUMMARY" "$EVIDENCE"
mkdir -p "$EVIDENCE"

# A fixed sleep can let a short benchmark finish before Instruments has
# attached. Register first, then wait for xctrace's Darwin notification so all
# measured requests are eligible to appear in the trace.
notifyutil -1 "$NOTIFICATION" >"$READY_FILE" &
READY_PID=$!
sleep 0.1
xcrun xctrace record \
  --template 'Metal System Trace' \
  --attach "$PID" \
  --time-limit "$DURATION" \
  --notify-tracing-started "$NOTIFICATION" \
  --output "$OUTPUT" &
TRACE_PID=$!

ready_polls=$((READY_TIMEOUT * 10))
while kill -0 "$READY_PID" 2>/dev/null; do
  if ! kill -0 "$TRACE_PID" 2>/dev/null; then
    wait "$TRACE_PID" || true
    fail "xctrace exited before recording became ready"
  fi
  if [ "$ready_polls" -le 0 ]; then
    fail "timed out after ${READY_TIMEOUT}s waiting for xctrace to start recording"
  fi
  ready_polls=$((ready_polls - 1))
  sleep 0.1
done
wait "$READY_PID" || fail "failed while waiting for the xctrace readiness notification"
READY_PID=

python3 "$ROOT/tools/bench_http.py" \
  --url "$URL" \
  --requests "$REQUESTS" \
  --concurrency "$CONCURRENCY" \
  --warmup 0 \
  --pid "$PID" \
  --output "$OUTPUT.benchmark.json" >/dev/null
wait "$TRACE_PID" || fail "xctrace recording failed"
TRACE_PID=

xcrun xctrace export --input "$OUTPUT" --toc --output "$TOC" >/dev/null \
  || fail "failed to export the trace table of contents"

export_table() {
  schema=$1
  filename=$2
  if ! xcrun xctrace export \
    --input "$OUTPUT" \
    --xpath "/trace-toc/run[@number=\"1\"]/data/table[@schema=\"$schema\"]" \
    --output "$EVIDENCE/$filename" >/dev/null; then
    echo "profile_metal: warning: could not export $schema" >&2
    rm -f "$EVIDENCE/$filename"
  fi
}

export_table metal-application-command-buffer-submissions submissions.xml
export_table metal-command-buffer-completed completed.xml
export_table metal-command-buffer-error errors.xml
export_table metal-gpu-intervals gpu-intervals.xml
export_table device-gpu-info device.xml

python3 "$ROOT/tools/summarize_metal_trace.py" \
  --evidence-dir "$EVIDENCE" \
  --benchmark "$OUTPUT.benchmark.json" \
  --trace "$OUTPUT" \
  --pid "$PID" \
  --output "$SUMMARY"

echo "trace: $OUTPUT"
echo "benchmark: $OUTPUT.benchmark.json"
echo "evidence: $EVIDENCE"
echo "summary: $SUMMARY"
echo "toc: $TOC"
