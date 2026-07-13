#!/bin/sh
set -eu

BASE=${MARIAN_MLX_HOME:-"$HOME/.local/share/marian-mlx"}
STATE=${MARIAN_MLX_STATE:-"$HOME/.local/state/marian-mlx"}
BIN_DIR=${MARIAN_MLX_BIN_DIR:-"$HOME/.local/bin"}
LABEL=io.github.malusama.marian-mlx
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"
PURGE=false
case "${1:-}" in
  "") ;;
  --purge) PURGE=true ;;
  *) echo "usage: $0 [--purge]" >&2; exit 2 ;;
esac
case "$BASE" in /*) ;; *) echo "MARIAN_MLX_HOME must be an absolute path" >&2; exit 1 ;; esac
case "$STATE" in /*) ;; *) echo "MARIAN_MLX_STATE must be an absolute path" >&2; exit 1 ;; esac
case "$BIN_DIR" in /*) ;; *) echo "MARIAN_MLX_BIN_DIR must be an absolute path" >&2; exit 1 ;; esac
case "$BASE" in
  ""|/|"$HOME") echo "refusing to remove unsafe MARIAN_MLX_HOME: $BASE" >&2; exit 1 ;;
esac
case "$STATE" in
  ""|/|"$HOME") echo "refusing to remove unsafe MARIAN_MLX_STATE: $STATE" >&2; exit 1 ;;
esac
case "$BIN_DIR" in
  ""|/) echo "refusing to remove unsafe MARIAN_MLX_BIN_DIR: $BIN_DIR" >&2; exit 1 ;;
esac

if [ -e "$BASE/config/legacy-label" ] || [ -L "$BASE/config/legacy-label" ] ||
   [ -e "$BASE/config/legacy-archive" ] || [ -L "$BASE/config/legacy-archive" ]; then
  if [ ! -x "$BIN_DIR/marian-mlxctl" ]; then
    echo "cannot restore the previous service: $BIN_DIR/marian-mlxctl is missing" >&2
    exit 1
  fi
  "$BIN_DIR/marian-mlxctl" restore-previous-service
fi

if launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
  NEW_PID=$(launchctl print "$DOMAIN/$LABEL" 2>/dev/null |
    awk '$1 == "pid" && $2 == "=" {print $3; exit}')
  launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || {
    echo "could not stop $LABEL; nothing was removed" >&2
    exit 1
  }
  attempt=0
  while { launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1 ||
          { [ -n "$NEW_PID" ] && kill -0 "$NEW_PID" >/dev/null 2>&1; }; } &&
        [ "$attempt" -lt 100 ]; do
    attempt=$((attempt + 1))
    sleep 0.1
  done
  if launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1 ||
     { [ -n "$NEW_PID" ] && kill -0 "$NEW_PID" >/dev/null 2>&1; }; then
    echo "service shutdown did not complete; nothing was removed" >&2
    exit 1
  fi
fi
rm -f "$PLIST" "$BIN_DIR/marian-mlxctl"
if [ "$PURGE" = true ]; then
  rm -rf "$BASE" "$STATE"
  echo "Marian MLX service, releases, model, cache, and logs removed."
else
  rm -rf "$BASE/current" "$BASE/previous" "$BASE/versions" "$BASE/tools" \
    "$BASE/config"
  rm -f "$BASE/uninstall-macos.sh"
  echo "Marian MLX removed; model and cache preserved under $BASE."
fi
