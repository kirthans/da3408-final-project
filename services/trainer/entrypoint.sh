#!/bin/sh
set -eu
mkdir -p "$HOME"
git config --global --add safe.directory /repo
exec "$@"
