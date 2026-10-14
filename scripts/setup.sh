#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
python3 -m venv .venv
source scripts/env.sh
python -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple pip==25.1.1
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1
bash scripts/finish_environment.sh
