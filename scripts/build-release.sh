#!/bin/sh
set -eu

ROOT=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

if [ -z "${MARIAN_MLX_BUILD_GIT_SHA:-}" ]; then
  MARIAN_MLX_BUILD_GIT_SHA=$(git rev-parse --short=12 HEAD 2>/dev/null || printf unknown)
  export MARIAN_MLX_BUILD_GIT_SHA
fi

REMAP="--remap-path-prefix=$ROOT=/workspace/marian-mlx --remap-path-prefix=$HOME=/home/build"
export RUSTFLAGS="${RUSTFLAGS:-} $REMAP"
PREFIX_MAP="-ffile-prefix-map=$ROOT=/workspace/marian-mlx -fmacro-prefix-map=$ROOT=/workspace/marian-mlx -ffile-prefix-map=$HOME=/home/build -fmacro-prefix-map=$HOME=/home/build"
export CFLAGS="${CFLAGS:-} $PREFIX_MAP"
export CXXFLAGS="${CXXFLAGS:-} $PREFIX_MAP"

if [ ! -f build/mlx-install/lib/libmlx.dylib ]; then
  echo "MLX is not built; run scripts/build-mlx.sh first" >&2
  exit 1
fi

HOST=$(rustc -vV | awk '/^host:/{print $2}')
if [ "$HOST" = aarch64-apple-darwin ]; then
  cargo build --locked --release -p marian-server --features mlx
else
  TOOLCHAIN=$(rustup toolchain list | awk '/aarch64-apple-darwin/{print $1}' | tail -1)
  if [ -z "$TOOLCHAIN" ]; then
    echo "Install a native toolchain: rustup toolchain install stable-aarch64-apple-darwin" >&2
    exit 1
  fi
  rustup run "$TOOLCHAIN" cargo build --locked --release -p marian-server --features mlx
fi

file target/release/marian-mlx-server
