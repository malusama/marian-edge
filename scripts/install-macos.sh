#!/bin/sh
set -eu

REPOSITORY=${MARIAN_MLX_REPOSITORY:-malusama/marian-mlx}
REQUESTED_VERSION=${MARIAN_MLX_VERSION:-latest}
if [ "${MARIAN_MLX_PORT+x}" = x ]; then
  PORT=$MARIAN_MLX_PORT
  PORT_WAS_EXPLICIT=true
else
  PORT=
  PORT_WAS_EXPLICIT=false
fi
if [ "${MARIAN_MLX_CORS_ORIGIN+x}" = x ]; then
  CORS_ORIGIN=$MARIAN_MLX_CORS_ORIGIN
  CORS_ORIGIN_WAS_EXPLICIT=true
else
  CORS_ORIGIN=
  CORS_ORIGIN_WAS_EXPLICIT=false
fi
BASE=${MARIAN_MLX_HOME:-"$HOME/.local/share/marian-mlx"}
STATE=${MARIAN_MLX_STATE:-"$HOME/.local/state/marian-mlx"}
BIN_DIR=${MARIAN_MLX_BIN_DIR:-"$HOME/.local/bin"}
LABEL=io.github.malusama.marian-mlx
LEGACY_LABEL=${MARIAN_MLX_LEGACY_LABEL:-}
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LEGACY_PLIST=
CONFIG_DIR="$BASE/config"
PORT_CONFIG="$CONFIG_DIR/port"
CORS_CONFIG="$CONFIG_DIR/cors-origin"
LEGACY_LABEL_CONFIG="$CONFIG_DIR/legacy-label"
LEGACY_ARCHIVE_CONFIG="$CONFIG_DIR/legacy-archive"
ASSET=marian-mlx-macos-arm64.tar.gz
UV_VERSION=0.11.28
UV_INSTALLER_SHA256=b7b3fe80cad1142a2a5794050b7db7b3291d1bac1423b0732571dd9366e8ca8b

say() { printf '%s\n' "marian-mlx: $*"; }
fail() { printf '%s\n' "marian-mlx: error: $*" >&2; exit 1; }

if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  fail "the native installer requires Apple Silicon macOS"
fi
MACOS_MAJOR=$(sw_vers -productVersion | awk -F. '{print $1}')
if [ "$MACOS_MAJOR" -lt 14 ]; then
  fail "macOS 14 or newer is required"
fi
case "$BASE" in
  /*) ;;
  *) fail "MARIAN_MLX_HOME must be an absolute path" ;;
esac
case "$STATE" in
  /*) ;;
  *) fail "MARIAN_MLX_STATE must be an absolute path" ;;
esac
case "$BIN_DIR" in
  /*) ;;
  *) fail "MARIAN_MLX_BIN_DIR must be an absolute path" ;;
esac
case "$BASE" in ""|/|"$HOME") fail "unsafe MARIAN_MLX_HOME: $BASE" ;; esac
case "$STATE" in ""|/|"$HOME") fail "unsafe MARIAN_MLX_STATE: $STATE" ;; esac
case "$BIN_DIR" in ""|/) fail "unsafe MARIAN_MLX_BIN_DIR: $BIN_DIR" ;; esac
if [ "$PORT_WAS_EXPLICIT" = false ]; then
  if [ -e "$PORT_CONFIG" ] || [ -L "$PORT_CONFIG" ]; then
    if [ ! -f "$PORT_CONFIG" ] || [ -L "$PORT_CONFIG" ] ||
       [ "$(awk 'END {print NR}' "$PORT_CONFIG")" -ne 1 ]; then
      fail "the saved service port is invalid"
    fi
    PORT=$(sed -n '1p' "$PORT_CONFIG")
  else
    PORT=3000
  fi
fi
case "$PORT" in
  ''|*[!0-9]*) fail "MARIAN_MLX_PORT must be a number" ;;
esac
if [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
  fail "MARIAN_MLX_PORT must be between 1 and 65535"
fi
if [ "$CORS_ORIGIN_WAS_EXPLICIT" = false ] &&
   { [ -e "$CORS_CONFIG" ] || [ -L "$CORS_CONFIG" ]; }; then
  if [ ! -f "$CORS_CONFIG" ] || [ -L "$CORS_CONFIG" ] ||
     [ "$(awk 'END {print NR}' "$CORS_CONFIG")" -ne 1 ]; then
    fail "the saved CORS origin is invalid"
  fi
  CORS_ORIGIN=$(sed -n '1p' "$CORS_CONFIG")
fi
if [ "$(printf '%s' "$CORS_ORIGIN" | wc -l | tr -d ' ')" -ne 0 ]; then
  fail "MARIAN_MLX_CORS_ORIGIN must be a single line"
fi
if [ -n "$LEGACY_LABEL" ]; then
  case "$LEGACY_LABEL" in
    *[!A-Za-z0-9._-]*) fail "MARIAN_MLX_LEGACY_LABEL contains an invalid character" ;;
  esac
  [ "$LEGACY_LABEL" != "$LABEL" ] || fail "MARIAN_MLX_LEGACY_LABEL must name a different service"
  LEGACY_PLIST="$HOME/Library/LaunchAgents/$LEGACY_LABEL.plist"
fi

USER_ID=$(id -u)
DOMAIN="gui/$USER_ID"
is_loaded() { launchctl print "$DOMAIN/$1" >/dev/null 2>&1; }
is_disabled() {
  launchctl print-disabled "$DOMAIN" 2>/dev/null |
    awk -v label="\"$1\"" '$1 == label && $2 == "=>" && $3 == "disabled" {found = 1} END {exit !found}'
}
port_is_listening() { lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; }
wait_for_port_free() {
  attempt=0
  while port_is_listening && [ "$attempt" -lt 100 ]; do
    attempt=$((attempt + 1))
    sleep 0.1
  done
  ! port_is_listening
}
NEW_WAS_LOADED=false
LEGACY_WAS_LOADED=false
NEW_WAS_DISABLED=false
LEGACY_WAS_DISABLED=false
LEGACY_PRESENT=false
is_loaded "$LABEL" && NEW_WAS_LOADED=true
is_disabled "$LABEL" && NEW_WAS_DISABLED=true
if [ -n "$LEGACY_LABEL" ] && is_loaded "$LEGACY_LABEL"; then
  LEGACY_WAS_LOADED=true
fi
if [ -n "$LEGACY_LABEL" ] && is_disabled "$LEGACY_LABEL"; then
  LEGACY_WAS_DISABLED=true
fi
if [ -n "$LEGACY_LABEL" ] && { [ -e "$LEGACY_PLIST" ] || [ -L "$LEGACY_PLIST" ]; }; then
  if [ ! -f "$LEGACY_PLIST" ] || [ -L "$LEGACY_PLIST" ]; then
    fail "the previous LaunchAgent plist must be a regular file"
  fi
  plutil -lint "$LEGACY_PLIST" >/dev/null || fail "the previous LaunchAgent plist is invalid"
  PLIST_LABEL=$(plutil -extract Label raw -o - "$LEGACY_PLIST" 2>/dev/null || true)
  [ "$PLIST_LABEL" = "$LEGACY_LABEL" ] || \
    fail "the previous LaunchAgent plist label does not match MARIAN_MLX_LEGACY_LABEL"
  LEGACY_PRESENT=true
elif [ "$LEGACY_WAS_LOADED" = true ]; then
  fail "the previous service is loaded but its LaunchAgent plist is missing"
fi
if [ "$LEGACY_PRESENT" = true ] && [ "$LEGACY_WAS_LOADED" = false ]; then
  fail "the previous LaunchAgent exists but is not loaded"
fi
if port_is_listening && \
   [ "$NEW_WAS_LOADED" = false ] && [ "$LEGACY_WAS_LOADED" = false ]; then
  fail "port $PORT is already used by an unrelated process"
fi

AVAILABLE_KB=$(df -Pk "$HOME" | awk 'NR==2 {print $4}')
if [ "${AVAILABLE_KB:-0}" -lt 750000 ]; then
  fail "at least 750 MB of free disk space is required for first install"
fi

TMP=$(mktemp -d "${TMPDIR:-/tmp}/marian-mlx-install.XXXXXX")
CUTOVER_STARTED=false
CUTOVER_COMMITTED=false
LEGACY_DISABLED=false
LEGACY_ARCHIVED=false
LEGACY_ARCHIVE=
OLD_CURRENT=
OLD_PREVIOUS=
CURRENT_WAS_LINK=false
PREVIOUS_WAS_LINK=false
CTL_WAS_PRESENT=false
UNINSTALL_WAS_PRESENT=false
PORT_CONFIG_WAS_PRESENT=false
CORS_CONFIG_WAS_PRESENT=false
LEGACY_LABEL_CONFIG_WAS_PRESENT=false
LEGACY_ARCHIVE_CONFIG_WAS_PRESENT=false

restore_link() {
  destination=$1
  was_link=$2
  old_target=$3
  rm -f "$destination.new" || return 1
  if [ "$was_link" = true ]; then
    ln -sfn "$old_target" "$destination.new" || return 1
    mv -h "$destination.new" "$destination" || return 1
  else
    rm -f "$destination" || return 1
  fi
}

new_service_is_ready() {
  is_loaded "$LABEL" &&
    curl --max-time 2 -fsS "http://127.0.0.1:$PORT/readyz" >/dev/null 2>&1 &&
    curl --max-time 2 -fsS "http://127.0.0.1:$PORT/info" 2>/dev/null |
      grep -Eq '"name"[[:space:]]*:[[:space:]]*"(metal|mlx)"'
}

legacy_service_is_ready() {
  is_loaded "$LEGACY_LABEL" &&
    curl --max-time 2 -fsS "http://127.0.0.1:$PORT/readyz" >/dev/null 2>&1
}

wait_for_original_service() {
  attempt=0
  while [ "$attempt" -lt 60 ]; do
    if [ "$NEW_WAS_LOADED" = true ] && new_service_is_ready; then
      return 0
    fi
    if [ "$NEW_WAS_LOADED" = false ] && [ "$LEGACY_WAS_LOADED" = true ] && \
       legacy_service_is_ready; then
      return 0
    fi
    attempt=$((attempt + 1))
    sleep 1
  done
  [ "$NEW_WAS_LOADED" = false ] && [ "$LEGACY_WAS_LOADED" = false ]
}

rollback_cutover() {
  set +e
  ROLLBACK_OK=true
  say "restoring the previous service"
  if is_loaded "$LABEL"; then
    launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || ROLLBACK_OK=false
  fi
  if [ "$LEGACY_WAS_LOADED" = false ] || ! is_loaded "$LEGACY_LABEL"; then
    wait_for_port_free || ROLLBACK_OK=false
  fi
  restore_link "$BASE/current" "$CURRENT_WAS_LINK" "$OLD_CURRENT" || ROLLBACK_OK=false
  restore_link "$BASE/previous" "$PREVIOUS_WAS_LINK" "$OLD_PREVIOUS" || ROLLBACK_OK=false
  if [ -f "$TMP/previous.plist" ]; then
    install -m 0644 "$TMP/previous.plist" "$PLIST" || ROLLBACK_OK=false
  else
    rm -f "$PLIST" || ROLLBACK_OK=false
  fi
  if [ "$CTL_WAS_PRESENT" = true ]; then
    install -m 0755 "$TMP/previous.marian-mlxctl" "$BIN_DIR/marian-mlxctl" || ROLLBACK_OK=false
  else
    rm -f "$BIN_DIR/marian-mlxctl" || ROLLBACK_OK=false
  fi
  if [ "$UNINSTALL_WAS_PRESENT" = true ]; then
    install -m 0755 "$TMP/previous.uninstall-macos.sh" "$BASE/uninstall-macos.sh" || \
      ROLLBACK_OK=false
  else
    rm -f "$BASE/uninstall-macos.sh" || ROLLBACK_OK=false
  fi
  if [ "$PORT_CONFIG_WAS_PRESENT" = true ]; then
    install -m 0644 "$TMP/previous.port" "$PORT_CONFIG" || ROLLBACK_OK=false
  else
    rm -f "$PORT_CONFIG" || ROLLBACK_OK=false
  fi
  if [ "$CORS_CONFIG_WAS_PRESENT" = true ]; then
    install -m 0644 "$TMP/previous.cors-origin" "$CORS_CONFIG" || ROLLBACK_OK=false
  else
    rm -f "$CORS_CONFIG" || ROLLBACK_OK=false
  fi
  if [ "$LEGACY_LABEL_CONFIG_WAS_PRESENT" = true ]; then
    install -m 0644 "$TMP/previous.legacy-label" "$LEGACY_LABEL_CONFIG" || ROLLBACK_OK=false
  else
    rm -f "$LEGACY_LABEL_CONFIG" || ROLLBACK_OK=false
  fi
  if [ "$LEGACY_ARCHIVE_CONFIG_WAS_PRESENT" = true ]; then
    install -m 0644 "$TMP/previous.legacy-archive" "$LEGACY_ARCHIVE_CONFIG" || \
      ROLLBACK_OK=false
  else
    rm -f "$LEGACY_ARCHIVE_CONFIG" || ROLLBACK_OK=false
  fi
  if [ "$LEGACY_ARCHIVED" = true ] && [ -f "$LEGACY_ARCHIVE" ]; then
    if [ -e "$LEGACY_PLIST" ] || [ -L "$LEGACY_PLIST" ]; then
      ROLLBACK_OK=false
      say "warning: could not restore the previous LaunchAgent plist"
    else
      mv "$LEGACY_ARCHIVE" "$LEGACY_PLIST" || ROLLBACK_OK=false
    fi
  fi
  if [ "$NEW_WAS_LOADED" = true ] && [ -f "$PLIST" ]; then
    launchctl enable "$DOMAIN/$LABEL" >/dev/null 2>&1 || ROLLBACK_OK=false
    if ! is_loaded "$LABEL"; then
      launchctl bootstrap "$DOMAIN" "$PLIST" >/dev/null 2>&1 || ROLLBACK_OK=false
    fi
    launchctl kickstart -k "$DOMAIN/$LABEL" >/dev/null 2>&1 || ROLLBACK_OK=false
  elif [ "$LEGACY_WAS_LOADED" = true ] && [ -f "$LEGACY_PLIST" ]; then
    launchctl enable "$DOMAIN/$LEGACY_LABEL" >/dev/null 2>&1 || ROLLBACK_OK=false
    if ! is_loaded "$LEGACY_LABEL"; then
      launchctl bootstrap "$DOMAIN" "$LEGACY_PLIST" >/dev/null 2>&1 || ROLLBACK_OK=false
    fi
    launchctl kickstart -k "$DOMAIN/$LEGACY_LABEL" >/dev/null 2>&1 || ROLLBACK_OK=false
  fi
  if ! wait_for_original_service; then
    ROLLBACK_OK=false
  fi
  if [ "$NEW_WAS_DISABLED" = true ]; then
    launchctl disable "$DOMAIN/$LABEL" >/dev/null 2>&1 || ROLLBACK_OK=false
  elif [ "$NEW_WAS_LOADED" = true ]; then
    launchctl enable "$DOMAIN/$LABEL" >/dev/null 2>&1 || ROLLBACK_OK=false
  fi
  if [ -n "$LEGACY_LABEL" ]; then
    if [ "$LEGACY_WAS_DISABLED" = true ]; then
      launchctl disable "$DOMAIN/$LEGACY_LABEL" >/dev/null 2>&1 || ROLLBACK_OK=false
    elif [ "$LEGACY_WAS_LOADED" = true ] || [ "$LEGACY_DISABLED" = true ]; then
      launchctl enable "$DOMAIN/$LEGACY_LABEL" >/dev/null 2>&1 || ROLLBACK_OK=false
    fi
  fi
  if [ "$ROLLBACK_OK" != true ]; then
    say "warning: automatic rollback needs manual verification"
    return 1
  fi
  return 0
}

cleanup() {
  status=$?
  trap - EXIT HUP INT TERM
  if [ "$CUTOVER_STARTED" = true ] && [ "$CUTOVER_COMMITTED" = false ]; then
    if ! rollback_cutover; then
      status=1
    fi
  fi
  rm -rf "$TMP"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' HUP INT TERM

mkdir -p "$BASE/versions" "$BASE/models" "$BASE/cache" "$BASE/tools" \
  "$BASE/legacy" "$CONFIG_DIR" "$STATE" "$BIN_DIR" "$(dirname -- "$PLIST")"

download() {
  destination=$1
  url=$2
  curl --proto '=https' --proto-redir '=https' --tlsv1.2 \
    --fail --location --retry 4 --retry-all-errors \
    --connect-timeout 15 --speed-limit 1024 --speed-time 60 --max-time 1800 \
    --output "$destination" "$url"
}

if [ "$REQUESTED_VERSION" = latest ]; then
  RELEASE_BASE="https://github.com/$REPOSITORY/releases/latest/download"
else
  RELEASE_BASE="https://github.com/$REPOSITORY/releases/download/$REQUESTED_VERSION"
fi

say "downloading the release bundle"
download "$TMP/$ASSET" "$RELEASE_BASE/$ASSET"
download "$TMP/SHA256SUMS" "$RELEASE_BASE/SHA256SUMS"
MATCH_COUNT=$(awk -v file="$ASSET" \
  '($2 == file || $2 == "*" file) && length($1) == 64 && $1 ~ /^[0-9A-Fa-f]+$/ {n++} END {print n + 0}' \
  "$TMP/SHA256SUMS")
[ "$MATCH_COUNT" -eq 1 ] || fail "release checksum must list $ASSET exactly once"
EXPECTED=$(awk -v file="$ASSET" \
  '($2 == file || $2 == "*" file) && length($1) == 64 && $1 ~ /^[0-9A-Fa-f]+$/ {print tolower($1)}' \
  "$TMP/SHA256SUMS")
ACTUAL=$(shasum -a 256 "$TMP/$ASSET" | awk '{print $1}')
[ "$ACTUAL" = "$EXPECTED" ] || fail "release archive checksum mismatch"

if tar -tzf "$TMP/$ASSET" | awk '
  /^\// {bad = 1}
  {
    count = split($0, part, "/")
    for (i = 1; i <= count; i++) if (part[i] == "..") bad = 1
  }
  END {exit bad ? 0 : 1}
'; then
  fail "release archive contains an unsafe path"
fi
TOP_LEVEL=$(tar -tzf "$TMP/$ASSET" | awk -F/ 'NF && $1 != "" {print $1}' | LC_ALL=C sort -u)
TOP_COUNT=$(printf '%s\n' "$TOP_LEVEL" | awk 'NF {n++} END {print n + 0}')
[ "$TOP_COUNT" -eq 1 ] || fail "release archive must contain one top-level directory"
case "$TOP_LEVEL" in
  marian-mlx-v*-macos-arm64) ;;
  *) fail "release archive layout is invalid" ;;
esac
tar -xzf "$TMP/$ASSET" -C "$TMP"
BUNDLE="$TMP/$TOP_LEVEL"
VERSION=$(sed -n '1p' "$BUNDLE/VERSION")
[ -n "$VERSION" ] || fail "release version is missing"
printf '%s\n' "$VERSION" | grep -Eq \
  '^[0-9]+\.[0-9]+\.[0-9]+([-.][0-9A-Za-z][0-9A-Za-z.-]*)?$' || \
  fail "release version is invalid"
[ "$TOP_LEVEL" = "marian-mlx-v$VERSION-macos-arm64" ] || \
  fail "release directory and VERSION do not match"

bundle_is_valid() {
  bundle=$1
  [ -d "$bundle" ] && \
  [ -x "$bundle/marian-mlx-server" ] && \
  [ -x "$bundle/scripts/prepare-enzh-model.sh" ] && \
  [ -x "$bundle/scripts/marian-mlxctl" ] && \
  [ -x "$bundle/scripts/uninstall-macos.sh" ] && \
  [ ! -L "$bundle/marian-mlx-server" ] && \
  [ ! -e "$bundle/libmlx.dylib" ] && \
  [ ! -e "$bundle/mlx.metallib" ] && \
  [ -f "$bundle/SHA256SUMS" ] && \
  ! find "$bundle" -type l -print -quit | grep -q . && \
  ! find "$bundle" \( -name '*.dylib' -o -name '*.metallib' \) -print -quit | grep -q . && \
  (cd "$bundle" && shasum -a 256 -c SHA256SUMS >/dev/null 2>&1) && \
  file "$bundle/marian-mlx-server" | grep -q arm64 && \
  codesign --verify --strict "$bundle/marian-mlx-server" >/dev/null 2>&1
}

bundle_is_valid "$BUNDLE" || fail "release bundle verification failed"
RELEASE_BASE_DIR="$BASE/versions/v$VERSION-$(printf '%s' "$ACTUAL" | cut -c1-12)"
RELEASE_DIR="$RELEASE_BASE_DIR"
if [ -e "$RELEASE_DIR" ] && ! bundle_is_valid "$RELEASE_DIR"; then
  RELEASE_DIR="$RELEASE_BASE_DIR-repair-$$"
fi

model_is_valid() {
  [ -f "$BASE/models/en-zh/manifest.json" ] && \
  grep -Eq '"format"[[:space:]]*:[[:space:]]*"marian-mlx\.transformer-ssru\.v1"' \
    "$BASE/models/en-zh/manifest.json" && \
  [ -f "$BASE/models/en-zh/model.fp32.safetensors" ] && \
  [ -f "$BASE/models/en-zh/source.spm" ] && \
  [ -f "$BASE/models/en-zh/target.spm" ] && \
  [ -f "$BASE/models/en-zh/shortlist.bin" ] && \
  [ "$(shasum -a 256 "$BASE/models/en-zh/model.fp32.safetensors" | awk '{print $1}')" = \
    e030217aa5d10fa719a938af10ac6138171687472d3142313284b35099679bd5 ] && \
  [ "$(shasum -a 256 "$BASE/models/en-zh/source.spm" | awk '{print $1}')" = \
    bd9b65504acc6d9726dd281f7defc2adb7c2c22d0688fe2f84697de25197c8c5 ] && \
  [ "$(shasum -a 256 "$BASE/models/en-zh/target.spm" | awk '{print $1}')" = \
    aded6993c36e440284d11cec3f6b8aef9c0e43188a772d80be342a713adf223d ] && \
  [ "$(shasum -a 256 "$BASE/models/en-zh/shortlist.bin" | awk '{print $1}')" = \
    8575d8daa10e2dbff316dcdf8e1ce475357bcc2c92bdc63b736a2d5add22f681 ]
}

if ! model_is_valid; then
  if command -v uv >/dev/null 2>&1; then
    UV_BIN=$(command -v uv)
  else
    say "installing pinned uv into $BASE/tools"
    UV_INSTALLER="$TMP/uv-installer.sh"
    download "$UV_INSTALLER" \
      "https://releases.astral.sh/github/uv/releases/download/$UV_VERSION/uv-installer.sh"
    UV_ACTUAL=$(shasum -a 256 "$UV_INSTALLER" | awk '{print $1}')
    [ "$UV_ACTUAL" = "$UV_INSTALLER_SHA256" ] || fail "uv installer checksum mismatch"
    UV_UNMANAGED_INSTALL="$BASE/tools" sh "$UV_INSTALLER"
    UV_BIN="$BASE/tools/uv"
    [ -x "$UV_BIN" ] || fail "uv installation did not produce an executable"
  fi
  say "downloading, verifying, and converting the en-to-zh model"
  MODEL_CACHE_DIR="$BASE/cache/mozilla-enzh" \
  UV_CACHE_DIR="$BASE/cache/uv" \
  UV_PYTHON_INSTALL_DIR="$BASE/cache/python" \
  UV_BIN="$UV_BIN" \
    "$BUNDLE/scripts/prepare-enzh-model.sh" "$BASE/models/en-zh"
  model_is_valid || fail "converted model verification failed"
else
  say "reusing the verified en-to-zh model"
fi

if [ ! -e "$RELEASE_DIR" ]; then
  STAGED_RELEASE="$BASE/versions/.v$VERSION.$$"
  rm -rf "$STAGED_RELEASE"
  mv "$BUNDLE" "$STAGED_RELEASE"
  mv "$STAGED_RELEASE" "$RELEASE_DIR"
fi
bundle_is_valid "$RELEASE_DIR" || fail "installed release verification failed"

if [ -L "$BASE/current" ]; then
  CURRENT_WAS_LINK=true
  OLD_CURRENT=$(readlink "$BASE/current")
elif [ -e "$BASE/current" ]; then
  fail "$BASE/current must be a symbolic link"
fi
if [ -L "$BASE/previous" ]; then
  PREVIOUS_WAS_LINK=true
  OLD_PREVIOUS=$(readlink "$BASE/previous")
elif [ -e "$BASE/previous" ]; then
  fail "$BASE/previous must be a symbolic link"
fi
[ -f "$PLIST" ] && cp -p "$PLIST" "$TMP/previous.plist"
if [ -f "$BIN_DIR/marian-mlxctl" ]; then
  CTL_WAS_PRESENT=true
  cp -p "$BIN_DIR/marian-mlxctl" "$TMP/previous.marian-mlxctl"
fi
if [ -f "$BASE/uninstall-macos.sh" ]; then
  UNINSTALL_WAS_PRESENT=true
  cp -p "$BASE/uninstall-macos.sh" "$TMP/previous.uninstall-macos.sh"
fi
if [ -f "$PORT_CONFIG" ]; then
  PORT_CONFIG_WAS_PRESENT=true
  cp -p "$PORT_CONFIG" "$TMP/previous.port"
fi
if [ -f "$CORS_CONFIG" ]; then
  CORS_CONFIG_WAS_PRESENT=true
  cp -p "$CORS_CONFIG" "$TMP/previous.cors-origin"
fi
if [ -f "$LEGACY_LABEL_CONFIG" ]; then
  LEGACY_LABEL_CONFIG_WAS_PRESENT=true
  cp -p "$LEGACY_LABEL_CONFIG" "$TMP/previous.legacy-label"
fi
if [ -f "$LEGACY_ARCHIVE_CONFIG" ]; then
  LEGACY_ARCHIVE_CONFIG_WAS_PRESENT=true
  cp -p "$LEGACY_ARCHIVE_CONFIG" "$TMP/previous.legacy-archive"
fi

CUTOVER_STARTED=true
if [ -n "$OLD_CURRENT" ] && [ "$OLD_CURRENT" != "$RELEASE_DIR" ]; then
  ln -sfn "$OLD_CURRENT" "$BASE/previous.new"
  mv -h "$BASE/previous.new" "$BASE/previous"
fi
ln -sfn "$RELEASE_DIR" "$BASE/current.new"
mv -h "$BASE/current.new" "$BASE/current"

xml_escape() {
  printf '%s' "$1" | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g'
}
sed_replacement() {
  printf '%s' "$1" | sed 's/[&|\\]/\\&/g'
}
PROGRAM=$(sed_replacement "$(xml_escape "$BASE/current/marian-mlx-server")")
MODEL_DIR=$(sed_replacement "$(xml_escape "$BASE/models/en-zh")")
STDOUT=$(sed_replacement "$(xml_escape "$STATE/server.log")")
STDERR=$(sed_replacement "$(xml_escape "$STATE/server.error.log")")
BIND=$(sed_replacement "$(xml_escape "127.0.0.1:$PORT")")
if [ -n "$CORS_ORIGIN" ]; then
  CORS_XML="<string>--cors-origin</string><string>$(xml_escape "$CORS_ORIGIN")</string>"
else
  CORS_XML=""
fi
CORS_XML=$(sed_replacement "$CORS_XML")

PLIST_NEW="$TMP/$LABEL.plist"
sed \
  -e "s|@PROGRAM@|$PROGRAM|g" \
  -e "s|@MODEL_DIR@|$MODEL_DIR|g" \
  -e "s|@STDOUT@|$STDOUT|g" \
  -e "s|@STDERR@|$STDERR|g" \
  -e "s|@BIND@|$BIND|g" \
  -e "s|@CORS_ARGUMENT@|$CORS_XML|g" \
  "$RELEASE_DIR/packaging/launchd/io.github.malusama.marian-mlx.plist" \
  > "$PLIST_NEW"
plutil -lint "$PLIST_NEW" >/dev/null || fail "generated LaunchAgent is invalid"

say "switching the LaunchAgent to v$VERSION"
if is_loaded "$LABEL"; then
  launchctl bootout "$DOMAIN/$LABEL" >/dev/null
fi
if [ -n "$LEGACY_LABEL" ] && is_loaded "$LEGACY_LABEL"; then
  launchctl bootout "$DOMAIN/$LEGACY_LABEL" >/dev/null
fi
if ! wait_for_port_free; then
  fail "port $PORT did not become available during the service switch"
fi
install -m 0644 "$PLIST_NEW" "$PLIST"
launchctl enable "$DOMAIN/$LABEL"
launchctl bootstrap "$DOMAIN" "$PLIST"
launchctl kickstart -k "$DOMAIN/$LABEL"

READY=false
attempt=0
while [ "$attempt" -lt 60 ]; do
  if is_loaded "$LABEL" && \
     curl --max-time 2 -fsS "http://127.0.0.1:$PORT/readyz" >/dev/null 2>&1; then
    INFO=$(curl --max-time 2 -fsS "http://127.0.0.1:$PORT/info" 2>/dev/null || true)
    if printf '%s' "$INFO" | grep -Eq '"name"[[:space:]]*:[[:space:]]*"(metal|mlx)"'; then
      READY=true
      break
    fi
  fi
  attempt=$((attempt + 1))
  sleep 1
done

if [ "$READY" != true ]; then
  fail "readiness check failed; see $STATE/server.error.log"
fi

install -m 0755 "$RELEASE_DIR/scripts/marian-mlxctl" "$BIN_DIR/marian-mlxctl"
install -m 0755 "$RELEASE_DIR/scripts/uninstall-macos.sh" "$BASE/uninstall-macos.sh"
printf '%s\n' "$PORT" > "$TMP/port"
install -m 0644 "$TMP/port" "$PORT_CONFIG"
if [ -n "$CORS_ORIGIN" ]; then
  printf '%s\n' "$CORS_ORIGIN" > "$TMP/cors-origin"
  install -m 0644 "$TMP/cors-origin" "$CORS_CONFIG"
else
  rm -f "$CORS_CONFIG"
fi
if [ "$LEGACY_PRESENT" = true ]; then
  LEGACY_DISABLED=true
  launchctl disable "$DOMAIN/$LEGACY_LABEL"
  LEGACY_ARCHIVE="$BASE/legacy/$LEGACY_LABEL.$(date +%Y%m%d%H%M%S).$$.plist"
  LEGACY_ARCHIVED=true
  mv "$LEGACY_PLIST" "$LEGACY_ARCHIVE"
  printf '%s\n' "$LEGACY_LABEL" > "$TMP/legacy-label"
  printf '%s\n' "$LEGACY_ARCHIVE" > "$TMP/legacy-archive"
  install -m 0644 "$TMP/legacy-label" "$LEGACY_LABEL_CONFIG"
  install -m 0644 "$TMP/legacy-archive" "$LEGACY_ARCHIVE_CONFIG"
fi
CUTOVER_COMMITTED=true
say "installed v$VERSION on http://127.0.0.1:$PORT"
say "runtime: $INFO"
say "control command: $BIN_DIR/marian-mlxctl status"
