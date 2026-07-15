#!/bin/sh
set -eu

ROOT=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

if [ -z "${MARIAN_EDGE_BUILD_GIT_SHA:-}" ]; then
  MARIAN_EDGE_BUILD_GIT_SHA=$(git rev-parse --short=12 HEAD 2>/dev/null || printf unknown)
  export MARIAN_EDGE_BUILD_GIT_SHA
fi

REMAP="--remap-path-prefix=$ROOT=/workspace/marian-edge --remap-path-prefix=$HOME=/home/build"
export RUSTFLAGS="${RUSTFLAGS:-} $REMAP"

HOST=$(rustc -vV | awk '/^host:/{print $2}')
if [ "$HOST" = aarch64-apple-darwin ]; then
  cargo build --locked --release -p marian-server --features metal
else
  TOOLCHAIN=$(rustup toolchain list | awk '/aarch64-apple-darwin/{print $1}' | tail -1)
  if [ -z "$TOOLCHAIN" ]; then
    echo "Install a native toolchain: rustup toolchain install stable-aarch64-apple-darwin" >&2
    exit 1
  fi
  rustup run "$TOOLCHAIN" cargo build --locked --release -p marian-server --features metal
fi

file target/release/marian-edge-server
