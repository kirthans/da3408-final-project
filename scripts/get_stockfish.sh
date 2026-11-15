#!/usr/bin/env bash
# Install the official pinned release without system package managers.
set -euo pipefail
cd "$(dirname "$0")/.."
build=x86-64
if grep -qw avx2 /proc/cpuinfo; then build=x86-64-avx2; fi
archive=$(mktemp /tmp/elo-stockfish.XXXXXX.tar)
extract=$(mktemp -d /tmp/elo-stockfish.XXXXXX)
trap 'rm -rf "$archive" "$extract"' EXIT
curl --fail --location --retry 3 \
  "https://github.com/official-stockfish/Stockfish/releases/download/sf_17.1/stockfish-ubuntu-${build}.tar" \
  --output "$archive"
tar -xf "$archive" -C "$extract"
mkdir -p tools/stockfish
install -m 755 "$extract/stockfish/stockfish-ubuntu-${build}" tools/stockfish/stockfish
printf 'uci\nquit\n' | tools/stockfish/stockfish
