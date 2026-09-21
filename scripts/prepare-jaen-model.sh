#!/bin/sh
set -eu

ROOT=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
PRECISION=${MARIAN_EDGE_MODEL_PRECISION:-fp32}
case "$PRECISION" in
  fp32) DEFAULT_OUTPUT="$ROOT/models/jaen" ;;
  *) echo "MARIAN_EDGE_MODEL_PRECISION must be fp32" >&2; exit 2 ;;
esac
OUTPUT=${1:-"$DEFAULT_OUTPUT"}
CACHE=${MODEL_CACHE_DIR:-"$ROOT/.cache/mozilla-jaen"}
UV_BIN=${UV_BIN:-uv}
PYTHON_VERSION=${MARIAN_EDGE_CONVERTER_PYTHON:-${MARIAN_MLX_CONVERTER_PYTHON:-3.12}}
STAGING="${OUTPUT}.staging.$$"
PREVIOUS="${OUTPUT}.previous.$$"
BASE='https://storage.googleapis.com/moz-fx-translations-data--303e-prod-translations-data/models/ja-en/cjk_retrain_base-memory_NLRJLD_pQFyrvgKtbie2nA'

mkdir -p "$CACHE" "$(dirname -- "$OUTPUT")"
rm -rf "$STAGING" "$PREVIOUS"
trap 'rm -rf "$STAGING" "$PREVIOUS"' EXIT HUP INT TERM

verify() {
  expected=$1
  file=$2
  actual=$(shasum -a 256 "$file" | awk '{print $1}')
  if [ "$actual" != "$expected" ]; then
    echo "checksum mismatch for $file: expected $expected, got $actual" >&2
    return 1
  fi
}

download() {
  url=$1
  destination=$2
  expected=$3
  if [ -s "$destination" ] && verify "$expected" "$destination"; then
    return
  fi
  rm -f "$destination" "$destination.part"
  curl --fail --location --retry 4 --retry-all-errors \
    --connect-timeout 15 --speed-limit 1024 --speed-time 60 --max-time 1800 \
    --output "$destination.part" "$url"
  verify "$expected" "$destination.part"
  mv "$destination.part" "$destination"
}

if [ "$PRECISION" = fp32 ]; then
  download "$BASE/student-finetuned/final.model.npz.best-chrf.npz" "$CACHE/model.npz" \
    0af34a5a2062b00929c723ad787e5b242f7030b75e1aaac08f67c566f1792939
fi
download "$BASE/exported/vocab.jaen.spm.gz" "$CACHE/source.spm.gz" \
  12d693f5055525d5cc1e133c8c1b8ed787c77b9bb797400d9a14382ac69c1236
download "$BASE/exported/vocab.jaen.spm.gz" "$CACHE/target.spm.gz" \
  12d693f5055525d5cc1e133c8c1b8ed787c77b9bb797400d9a14382ac69c1236
download "$BASE/exported/lex.50.50.jaen.s2t.bin.gz" "$CACHE/shortlist.bin.gz" \
  438152f5ccd982edb43e88ef51305e3ae7c7b66ee5c20a8fa425e9f1822f9b9b


gzip -dc "$CACHE/source.spm.gz" > "$CACHE/source.spm.part"
mv "$CACHE/source.spm.part" "$CACHE/source.spm"
gzip -dc "$CACHE/target.spm.gz" > "$CACHE/target.spm.part"
mv "$CACHE/target.spm.part" "$CACHE/target.spm"
gzip -dc "$CACHE/shortlist.bin.gz" > "$CACHE/shortlist.bin.part"
mv "$CACHE/shortlist.bin.part" "$CACHE/shortlist.bin"

verify 5cb217758bae05877bb3f0c2f612e4e7c1e4cb03c10db11f4a47098d7ae62919 "$CACHE/source.spm"
verify 5cb217758bae05877bb3f0c2f612e4e7c1e4cb03c10db11f4a47098d7ae62919 "$CACHE/target.spm"
verify 525f412f0d210536c2933c78ae395fa0bf2b5ee6cc5dda61ebc2e79410ebaee4 "$CACHE/shortlist.bin"

  "$UV_BIN" run --isolated --python "$PYTHON_VERSION" \
    --with numpy==2.5.1 --with safetensors==0.8.0 \
    python "$ROOT/tools/convert_marian.py" \
    --model "$CACHE/model.npz" \
    --source-vocab "$CACHE/source.spm" \
    --target-vocab "$CACHE/target.spm" \
    --shortlist "$CACHE/shortlist.bin" \
    --output "$STAGING" \
    --model-id mozilla-firefox-translations-ja-en-base-memory-3.1 \
    --source-lang ja --target-lang en --allow-unverified-model \
    --force

  [ -s "$STAGING/model.fp32.safetensors" ] || {
    echo "converted FP32 weights are missing" >&2
    exit 1
  }
grep -Eq '"format"[[:space:]]*:[[:space:]]*"marian-edge\.transformer-ssru\.v1"' \
  "$STAGING/manifest.json" || {
  echo "converted model manifest is invalid" >&2
  exit 1
}

if [ -d "$OUTPUT" ]; then
  mv "$OUTPUT" "$PREVIOUS"
fi
mv "$STAGING" "$OUTPUT"
rm -rf "$PREVIOUS"
