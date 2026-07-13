#!/bin/sh
set -eu

ROOT=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
CACHE=${MLX_DEPS_DIR:-"$ROOT/.cache/mlx-deps"}
ARCHIVES="$CACHE/archives"
mkdir -p "$ARCHIVES"

download_and_verify() {
  name=$1
  url=$2
  expected=$3
  destination="$ARCHIVES/$name"
  if [ ! -s "$destination" ]; then
    curl --fail --location --retry 4 --retry-all-errors \
      --connect-timeout 15 --speed-limit 1024 --speed-time 60 --max-time 1800 \
      --output "$destination.part" "$url"
    mv "$destination.part" "$destination"
  fi
  actual=$(shasum -a 256 "$destination" | awk '{print $1}')
  if [ "$actual" != "$expected" ]; then
    rm -f "$destination"
    echo "checksum mismatch for $name: expected $expected, got $actual" >&2
    exit 1
  fi
}

download_and_verify \
  metal-cpp_26.zip \
  https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip \
  4df3c078b9aadcb516212e9cb03004cbc5ce9a3e9c068fa3144d021db585a3a4
download_and_verify \
  json-3.11.3.tar.xz \
  https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz \
  d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
download_and_verify \
  fmt-12.1.0.tar.gz \
  https://github.com/fmtlib/fmt/archive/refs/tags/12.1.0.tar.gz \
  ea7de4299689e12b6dddd392f9896f08fb0777ac7168897a244a6d6085043fea

if [ ! -f "$CACHE/metal-cpp/Metal/Metal.hpp" ]; then
  rm -rf "$CACHE/metal-cpp" "$CACHE/.metal-cpp"
  mkdir -p "$CACHE/.metal-cpp"
  unzip -q "$ARCHIVES/metal-cpp_26.zip" -d "$CACHE/.metal-cpp"
  mv "$CACHE/.metal-cpp/metal-cpp" "$CACHE/metal-cpp"
  rmdir "$CACHE/.metal-cpp"
fi

if [ ! -f "$CACHE/json/CMakeLists.txt" ]; then
  rm -rf "$CACHE/json" "$CACHE/.json"
  mkdir -p "$CACHE/.json"
  tar -xJf "$ARCHIVES/json-3.11.3.tar.xz" -C "$CACHE/.json"
  mv "$CACHE/.json/json" "$CACHE/json"
  rmdir "$CACHE/.json"
fi

if [ ! -f "$CACHE/fmt/CMakeLists.txt" ]; then
  rm -rf "$CACHE/fmt" "$CACHE/.fmt"
  mkdir -p "$CACHE/.fmt"
  tar -xzf "$ARCHIVES/fmt-12.1.0.tar.gz" -C "$CACHE/.fmt"
  mv "$CACHE/.fmt/fmt-12.1.0" "$CACHE/fmt"
  rmdir "$CACHE/.fmt"
fi

echo "$CACHE"
