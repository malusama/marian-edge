#!/bin/sh
set -eu

ROOT=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
SOURCE="$ROOT/third_party/mlx"
BUILD="$ROOT/build/mlx"
PREFIX="$ROOT/build/mlx-install"
CMAKE_BIN=${CMAKE_BIN:-cmake}
PREFIX_MAP="-ffile-prefix-map=$ROOT=/workspace/marian-mlx -fmacro-prefix-map=$ROOT=/workspace/marian-mlx"
export CFLAGS="${CFLAGS:-} $PREFIX_MAP"
# MLX uses MLX_METAL_PATH both as a build output and as a compiled fallback.
# Keep the output absolute for CMake, then replace only the compiled fallback
# with a relocatable path. The service normally supplies the explicit path.
METAL_FALLBACK='-UMETAL_PATH -DMETAL_PATH=\"./mlx.metallib\"'
export CXXFLAGS="${CXXFLAGS:-} $PREFIX_MAP $METAL_FALLBACK"

# Intel Homebrew can appear first in PATH on older migrated Macs. An x86_64
# CMake reports the wrong host processor even when `-arch arm64` is supplied.
if [ -x /opt/homebrew/bin/cmake ]; then
  CMAKE_BIN=/opt/homebrew/bin/cmake
fi

if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  echo "MLX requires native Apple Silicon macOS" >&2
  exit 1
fi
if [ ! -f "$SOURCE/CMakeLists.txt" ]; then
  echo "MLX submodule is missing; run: git submodule update --init --recursive" >&2
  exit 1
fi

# MLX normally downloads these CMake dependencies during configure. Fetch and
# verify them ourselves, then force FetchContent into offline mode so a release
# build cannot drift when an upstream tag or archive changes.
DEPS=$("$ROOT/scripts/fetch-mlx-deps.sh")
mkdir -p "$BUILD/metal"

"$CMAKE_BIN" -S "$SOURCE" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
  -DCMAKE_C_FLAGS="$CFLAGS" \
  -DCMAKE_CXX_FLAGS="$CXXFLAGS" \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_INSTALL_NAME_DIR=@rpath \
  -DBUILD_SHARED_LIBS=ON \
  -DMLX_BUILD_METAL=ON \
  -DMLX_BUILD_CPU=ON \
  -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  -DMLX_BUILD_GGUF=OFF \
  -DMLX_BUILD_SAFETENSORS=ON \
  -DMLX_METAL_JIT=OFF \
  -DMLX_METAL_PATH="$BUILD/metal" \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
  -DFETCHCONTENT_SOURCE_DIR_METAL_CPP="$DEPS/metal-cpp" \
  -DFETCHCONTENT_SOURCE_DIR_JSON="$DEPS/json" \
  -DFETCHCONTENT_SOURCE_DIR_FMT="$DEPS/fmt"

"$CMAKE_BIN" --build "$BUILD" --parallel "$(sysctl -n hw.logicalcpu)"
"$CMAKE_BIN" --install "$BUILD"

echo "MLX installed in $PREFIX"
