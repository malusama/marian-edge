#!/bin/sh
set -eu

ROOT=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
OUTPUT=${1:-"$ROOT/models/enzh"}
CACHE=${MODEL_CACHE_DIR:-"$ROOT/.cache/mozilla-enzh"}
UV_BIN=${UV_BIN:-uv}
PYTHON_VERSION=${MARIAN_MLX_CONVERTER_PYTHON:-3.12}
STAGING="${OUTPUT}.staging.$$"
PREVIOUS="${OUTPUT}.previous.$$"
BASE='https://storage.googleapis.com/moz-fx-translations-data--303e-prod-translations-data/models/en-zh/llmaat_finetune10M_qe8_f2_ByQcSxGXQRqGi-UTxYE43g'

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

download "$BASE/student-finetuned/final.model.npz.best-chrf.npz" "$CACHE/model.npz" \
  9604368d0fb19aa431a82824cedd92205a68512b89086cbe8c4d8bd1585a8950
download "$BASE/exported/srcvocab.enzh.spm.gz" "$CACHE/source.spm.gz" \
  7846e3c236388390f4e5d321f8413d67f34c1bab5f066165eeb673bfd07607cc
download "$BASE/exported/trgvocab.enzh.spm.gz" "$CACHE/target.spm.gz" \
  4d641ce165b1f8478ee2ffb5149d2d46fab3779dc8fa1e9b97f9af1d2206c091
download "$BASE/exported/lex.50.50.enzh.s2t.bin.gz" "$CACHE/shortlist.bin.gz" \
  806f75821c0b838f4a8f4afe5bab3db8289cb7e5187753ba04c3bceadd75687a

gzip -dc "$CACHE/source.spm.gz" > "$CACHE/source.spm.part"
mv "$CACHE/source.spm.part" "$CACHE/source.spm"
gzip -dc "$CACHE/target.spm.gz" > "$CACHE/target.spm.part"
mv "$CACHE/target.spm.part" "$CACHE/target.spm"
gzip -dc "$CACHE/shortlist.bin.gz" > "$CACHE/shortlist.bin.part"
mv "$CACHE/shortlist.bin.part" "$CACHE/shortlist.bin"

verify bd9b65504acc6d9726dd281f7defc2adb7c2c22d0688fe2f84697de25197c8c5 "$CACHE/source.spm"
verify aded6993c36e440284d11cec3f6b8aef9c0e43188a772d80be342a713adf223d "$CACHE/target.spm"
verify 8575d8daa10e2dbff316dcdf8e1ce475357bcc2c92bdc63b736a2d5add22f681 "$CACHE/shortlist.bin"

"$UV_BIN" run --isolated --python "$PYTHON_VERSION" \
  --with numpy==2.5.1 --with safetensors==0.8.0 \
  python "$ROOT/tools/convert_marian.py" \
  --model "$CACHE/model.npz" \
  --source-vocab "$CACHE/source.spm" \
  --target-vocab "$CACHE/target.spm" \
  --shortlist "$CACHE/shortlist.bin" \
  --output "$STAGING" \
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
