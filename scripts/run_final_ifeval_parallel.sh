#!/usr/bin/env bash
set -euo pipefail
cd /ssd/lzx/mopd
mopd_infer/bin/python -u scripts/generate_benchmark_vllm.py --model /ssd/lzx/mopd/models/Qwen3-4B-Base --data /ssd/lzx/mopd/data/instruction/ifeval.jsonl --device 4 --max-seqs 32 --out /ssd/lzx/mopd/runs/full-eval/student-final-ifeval541 --adapter /ssd/lzx/mopd/runs/20261014-dual-domain-batch32-mgpu-opd-b8/checkpoint-0064
mopd_train/bin/python -u scripts/grade_benchmark_outputs.py --data /ssd/lzx/mopd/data/instruction/ifeval.jsonl --out /ssd/lzx/mopd/runs/full-eval/student-final-ifeval541
