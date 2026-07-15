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
BUNDLE="$DIST/marian-edge-v$VERSION-macos-arm64"
ARCHIVE="$DIST/marian-edge-macos-arm64.tar.gz"

printf '%s\n' "$VERSION" | grep -Eq \
  '^[0-9]+\.[0-9]+\.[0-9]+([-.][0-9A-Za-z][0-9A-Za-z.-]*)?$' || {
  echo "VERSION must be a semantic version without a leading v" >&2
  exit 2
}

"$ROOT/scripts/build-release.sh"
rm -rf "$BUNDLE"
mkdir -p "$BUNDLE/scripts" "$BUNDLE/tools" \
  "$BUNDLE/packaging/launchd"
install -m 0755 "$ROOT/target/release/marian-edge-server" "$BUNDLE/marian-edge-server"
install -m 0755 "$ROOT/scripts/prepare-enzh-model.sh" "$BUNDLE/scripts/prepare-enzh-model.sh"
install -m 0755 "$ROOT/scripts/marian-edgectl" "$BUNDLE/scripts/marian-edgectl"
install -m 0755 "$ROOT/scripts/uninstall-macos.sh" "$BUNDLE/scripts/uninstall-macos.sh"
install -m 0755 "$ROOT/tools/convert_marian.py" "$BUNDLE/tools/convert_marian.py"
install -m 0644 \
  "$ROOT/packaging/launchd/io.github.malusama.marian-edge.plist" \
  "$BUNDLE/packaging/launchd/io.github.malusama.marian-edge.plist"
install -m 0644 "$ROOT/LICENSE" "$BUNDLE/LICENSE"
install -m 0644 "$ROOT/LICENSE-APACHE-2.0" "$BUNDLE/LICENSE-APACHE-2.0"
install -m 0644 "$ROOT/THIRD_PARTY_NOTICES.md" "$BUNDLE/THIRD_PARTY_NOTICES.md"
printf '%s\n' "$VERSION" > "$BUNDLE/VERSION"

if [ "$(file -b "$BUNDLE/marian-edge-server")" = "" ] || \
   ! file "$BUNDLE/marian-edge-server" | grep -q 'arm64'; then
  echo "release executable is not arm64" >&2
  exit 1
fi
DYNAMIC_DEPENDENCIES=$(otool -L "$BUNDLE/marian-edge-server" | awk 'NR > 1 {print $1}')
if printf '%s\n' "$DYNAMIC_DEPENDENCIES" | \
   grep -Ev '^(/usr/lib/|/System/Library/)' | grep -q .; then
  echo "release executable has a non-system dynamic dependency" >&2
  exit 1
fi
if printf '%s\n' "$DYNAMIC_DEPENDENCIES" | \
   grep -Eiq '(^|/)(libc\+\+|libstdc\+\+|libsentencepiece)([^/]*)$'; then
  echo "direct Metal release must not link a C++ runtime or native SentencePiece" >&2
  exit 1
fi
if find "$BUNDLE" \( -name '*.dylib' -o -name '*.metallib' \) -print -quit | grep -q .; then
  echo "direct Metal release must not contain bundled dylib or metallib files" >&2
  exit 1
fi
if find "$BUNDLE" \( -iname '*sentencepiece*' -o -iname 'libtokenizer*.so' -o \
   -iname 'libtokenizer*.dylib' -o -iname 'libtokenizer*.a' \) \
   -print -quit | grep -q .; then
  echo "direct Metal release must not contain a bundled native tokenizer" >&2
  exit 1
fi

# CI uses ad-hoc signing. Release builders may provide a Developer ID identity.
CODESIGN_IDENTITY=${CODESIGN_IDENTITY:--}
codesign --force --sign "$CODESIGN_IDENTITY" "$BUNDLE/marian-edge-server"
codesign --verify --strict "$BUNDLE/marian-edge-server"
if strings "$BUNDLE/marian-edge-server" | grep -E '/Users/|/home/runner/' >/dev/null; then
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
