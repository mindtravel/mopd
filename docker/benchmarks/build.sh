#!/usr/bin/env bash
# Build mopd-bench:<tag> on para5090 from the pinned native deployments (/srv/ifbench, /srv/automationbench;
# moved from /ssd on 2026-10-24). IFBench keeps .git: its wrapper records the upstream commit.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TAG="${1:-latest}"
CTX="$(mktemp -d /ssd/lzx/mopd/tmp/bench-image.XXXX)"
trap 'rm -rf "$CTX"' EXIT
cp "$HERE/Dockerfile" "$HERE/entrypoint.sh" "$CTX/"
cp /srv/ifbench/runtime/uv "$CTX/uv"
mkdir -p "$CTX/ifbench" "$CTX/automationbench"
rsync -a --exclude .venv --exclude .pytest_cache /srv/ifbench/repo/ "$CTX/ifbench/repo/"
rsync -a /srv/ifbench/scripts/ "$CTX/ifbench/scripts/"
rsync -a /srv/ifbench/cache/nltk_data/ "$CTX/ifbench/nltk_data/"
rsync -a --exclude .git --exclude .venv --exclude .pytest_cache /srv/automationbench/repo/ "$CTX/automationbench/repo/"
docker build --network host -t "mopd-bench:$TAG" "$CTX"
docker run --rm --network host "mopd-bench:$TAG" selftest  # bridge networking is broken on this host
