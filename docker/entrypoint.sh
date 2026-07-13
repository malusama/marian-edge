#!/bin/sh
set -eu

/usr/local/bin/marian-mlx-prepare-model
exec /usr/local/bin/marian-mlx-server "$@"
