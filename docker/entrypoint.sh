#!/bin/sh
set -eu

/usr/local/bin/marian-edge-prepare-model
exec /usr/local/bin/marian-edge-server "$@"
