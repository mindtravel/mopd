#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/env.sh"
cd "$MOPD_ROOT"
python -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
python -m pip check
python -m pip freeze > requirements.lock.txt
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list()); assert torch.version.cuda == "12.8"'
torchrun --standalone --nproc_per_node=8 scripts/check_environment.py > logs/gpu-smoke.log 2>&1
printf 'PASS: base dependencies and eight-GPU smoke checks\n' > logs/environment-status.txt
