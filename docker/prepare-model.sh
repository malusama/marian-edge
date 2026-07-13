#!/bin/sh
set -eu

MODEL_DIR=${MARIAN_MLX_MODEL_DIR:-/models/en-zh}
BASE_URL=https://storage.googleapis.com/moz-fx-translations-data--303e-prod-translations-data/models/en-zh/llmaat_finetune10M_qe8_f2_ByQcSxGXQRqGi-UTxYE43g/exported
mkdir -p "$MODEL_DIR"
exec 9>"$MODEL_DIR/.prepare.lock"
flock -x -w 1800 9 || {
  echo "marian-mlx: timed out waiting for the model preparation lock" >&2
  exit 1
}

verify() {
  expected=$1
  file=$2
  [ -f "$file" ] && [ "$(sha256sum "$file" | awk '{print $1}')" = "$expected" ]
}

download_gzip() {
  remote=$1
  gzip_hash=$2
  output=$3
  output_hash=$4
  destination="$MODEL_DIR/$output"
  if verify "$output_hash" "$destination"; then
    return
  fi
  rm -f "$destination" "$destination.part" "$destination.gz.part"
  echo "marian-mlx: downloading Mozilla model artifact $remote"
  curl --fail --location --retry 4 --retry-all-errors \
    --connect-timeout 15 --speed-limit 1024 --speed-time 60 --max-time 1800 \
    --output "$destination.gz.part" "$BASE_URL/$remote"
  verify "$gzip_hash" "$destination.gz.part" || {
    rm -f "$destination.gz.part"
    echo "marian-mlx: checksum mismatch for $remote" >&2
    exit 1
  }
  gzip -dc "$destination.gz.part" > "$destination.part"
  verify "$output_hash" "$destination.part" || {
    rm -f "$destination.part" "$destination.gz.part"
    echo "marian-mlx: uncompressed checksum mismatch for $remote" >&2
    exit 1
  }
  mv "$destination.part" "$destination"
  rm -f "$destination.gz.part"
}

download_gzip \
  model.enzh.intgemm.alphas.bin.gz \
  7f255403b3bb2502f08ac4d5ca397a8a5a13f899d2f2e987a4934e089d241d16 \
  model.enzh.intgemm.alphas.bin \
  4e5accc141373565ddc8fa1565bceaa8d0c3482a82cab8131c719ebcc6c2157c
download_gzip \
  srcvocab.enzh.spm.gz \
  7846e3c236388390f4e5d321f8413d67f34c1bab5f066165eeb673bfd07607cc \
  source.spm \
  bd9b65504acc6d9726dd281f7defc2adb7c2c22d0688fe2f84697de25197c8c5
download_gzip \
  trgvocab.enzh.spm.gz \
  4d641ce165b1f8478ee2ffb5149d2d46fab3779dc8fa1e9b97f9af1d2206c091 \
  target.spm \
  aded6993c36e440284d11cec3f6b8aef9c0e43188a772d80be342a713adf223d
download_gzip \
  lex.50.50.enzh.s2t.bin.gz \
  806f75821c0b838f4a8f4afe5bab3db8289cb7e5187753ba04c3bceadd75687a \
  shortlist.bin \
  8575d8daa10e2dbff316dcdf8e1ce475357bcc2c92bdc63b736a2d5add22f681

CONFIG="$MODEL_DIR/bergamot.yml"
cat > "$CONFIG.part" <<EOF
models:
  - $MODEL_DIR/model.enzh.intgemm.alphas.bin
vocabs:
  - $MODEL_DIR/source.spm
  - $MODEL_DIR/target.spm
shortlist:
  - $MODEL_DIR/shortlist.bin
  - false
beam-size: 1
normalize: 1.0
word-penalty: 0
max-length-break: 128
mini-batch-words: 1024
workspace: 128
max-length-factor: 2.0
skip-cost: true
quiet: true
quiet-translation: true
gemm-precision: int8shiftAlphaAll
alignment: soft
EOF
mv "$CONFIG.part" "$CONFIG"
