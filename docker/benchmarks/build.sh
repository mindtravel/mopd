#!/usr/bin/env bash
# Build mopd-bench:<tag> on para5090 from the pinned native deployments (/ssd/ifbench, /ssd/automationbench).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TAG="${1:-latest}"
CTX="$(mktemp -d /ssd/lzx/mopd/tmp/bench-image.XXXX)"
trap 'rm -rf "$CTX"' EXIT
cp "$HERE/Dockerfile" "$HERE/entrypoint.sh" "$CTX/"
cp /ssd/ifbench/runtime/uv "$CTX/uv"
mkdir -p "$CTX/ifbench" "$CTX/automationbench"
rsync -a --exclude .git --exclude .venv --exclude .pytest_cache /ssd/ifbench/repo/ "$CTX/ifbench/repo/"
rsync -a /ssd/ifbench/scripts/ "$CTX/ifbench/scripts/"
rsync -a /ssd/ifbench/cache/nltk_data/ "$CTX/ifbench/nltk_data/"
rsync -a --exclude .git --exclude .venv --exclude .pytest_cache /ssd/automationbench/repo/ "$CTX/automationbench/repo/"
docker build --network host -t "mopd-bench:$TAG" "$CTX"
docker run --rm "mopd-bench:$TAG" selftest
