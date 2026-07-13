#!/bin/sh
set -eu

ROOT=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
WORKSPACE_VERSION=$(awk '
  /^\[workspace.package\]$/ {inside = 1; next}
  /^\[/ {inside = 0}
  inside && /^version = / {
    gsub(/^[^"]*"|".*$/, "")
    print
    exit
  }
' "$ROOT/Cargo.toml")
VERSION=${VERSION:-$WORKSPACE_VERSION}
DIST=${1:-"$ROOT/dist"}
BUNDLE="$DIST/marian-mlx-v$VERSION-macos-arm64"
ARCHIVE="$DIST/marian-mlx-macos-arm64.tar.gz"

printf '%s\n' "$VERSION" | grep -Eq \
  '^[0-9]+\.[0-9]+\.[0-9]+([-.][0-9A-Za-z][0-9A-Za-z.-]*)?$' || {
  echo "VERSION must be a semantic version without a leading v" >&2
  exit 2
}

"$ROOT/scripts/build-release.sh"
rm -rf "$BUNDLE"
mkdir -p "$BUNDLE/scripts" "$BUNDLE/tools" "$BUNDLE/licenses" \
  "$BUNDLE/packaging/launchd"
install -m 0755 "$ROOT/target/release/marian-mlx-server" "$BUNDLE/marian-mlx-server"
install -m 0755 "$ROOT/build/mlx-install/lib/libmlx.dylib" "$BUNDLE/libmlx.dylib"
install -m 0644 "$ROOT/build/mlx-install/lib/mlx.metallib" "$BUNDLE/mlx.metallib"
install -m 0755 "$ROOT/scripts/prepare-enzh-model.sh" "$BUNDLE/scripts/prepare-enzh-model.sh"
install -m 0755 "$ROOT/scripts/marian-mlxctl" "$BUNDLE/scripts/marian-mlxctl"
install -m 0755 "$ROOT/scripts/uninstall-macos.sh" "$BUNDLE/scripts/uninstall-macos.sh"
install -m 0755 "$ROOT/tools/convert_marian.py" "$BUNDLE/tools/convert_marian.py"
install -m 0644 \
  "$ROOT/packaging/launchd/io.github.malusama.marian-mlx.plist" \
  "$BUNDLE/packaging/launchd/io.github.malusama.marian-mlx.plist"
install -m 0644 "$ROOT/LICENSE" "$BUNDLE/LICENSE"
install -m 0644 "$ROOT/THIRD_PARTY_NOTICES.md" "$BUNDLE/THIRD_PARTY_NOTICES.md"
install -m 0644 "$ROOT/third_party/mlx/LICENSE" "$BUNDLE/licenses/MLX-LICENSE"
install -m 0644 "$ROOT/third_party/mlx/ACKNOWLEDGMENTS.md" "$BUNDLE/licenses/MLX-ACKNOWLEDGMENTS.md"
printf '%s\n' "$VERSION" > "$BUNDLE/VERSION"

if [ "$(file -b "$BUNDLE/marian-mlx-server")" = "" ] || \
   ! file "$BUNDLE/marian-mlx-server" | grep -q 'arm64'; then
  echo "release executable is not arm64" >&2
  exit 1
fi
if ! file "$BUNDLE/libmlx.dylib" | grep -q 'arm64'; then
  echo "MLX library is not arm64" >&2
  exit 1
fi
[ -s "$BUNDLE/mlx.metallib" ] || {
  echo "Metal library is empty" >&2
  exit 1
}
otool -L "$BUNDLE/marian-mlx-server" | grep -q '@rpath/libmlx.dylib'
RPATHS=$(otool -l "$BUNDLE/marian-mlx-server" | awk '
  $1 == "cmd" && $2 == "LC_RPATH" {getline; getline; print $2}
')
[ "$RPATHS" = "@executable_path" ] || {
  echo "release executable must use only @executable_path for MLX" >&2
  exit 1
}

# CI uses ad-hoc signing. Release builders may provide a Developer ID identity.
CODESIGN_IDENTITY=${CODESIGN_IDENTITY:--}
codesign --force --sign "$CODESIGN_IDENTITY" "$BUNDLE/libmlx.dylib"
codesign --force --sign "$CODESIGN_IDENTITY" "$BUNDLE/marian-mlx-server"
codesign --verify --strict "$BUNDLE/libmlx.dylib"
codesign --verify --strict "$BUNDLE/marian-mlx-server"
if strings "$BUNDLE/marian-mlx-server" "$BUNDLE/libmlx.dylib" \
   "$BUNDLE/mlx.metallib" | \
   grep -E '/Users/|/home/runner/' >/dev/null; then
  echo "release bundle contains a developer-machine path" >&2
  exit 1
fi

(
  cd "$BUNDLE"
  find . -type f ! -name SHA256SUMS -print | LC_ALL=C sort | \
    while IFS= read -r file; do
      shasum -a 256 "$file"
    done
) > "$BUNDLE/SHA256SUMS"
(cd "$BUNDLE" && shasum -a 256 -c SHA256SUMS >/dev/null)

rm -f "$ARCHIVE"
COPYFILE_DISABLE=1 tar -C "$DIST" -czf "$ARCHIVE" "$(basename "$BUNDLE")"
(cd "$DIST" && shasum -a 256 "$(basename "$ARCHIVE")" > SHA256SUMS)

echo "Bundle written to $BUNDLE"
echo "Archive written to $ARCHIVE"
echo "Run with MARIAN_MLX_METALLIB=$BUNDLE/mlx.metallib"
