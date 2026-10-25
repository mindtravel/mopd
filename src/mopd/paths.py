"""Every location the project depends on, in one place. MOPD_ROOT (exported by scripts/env.sh)
overrides the project root; everything else derives from it."""

import os
from pathlib import Path

ROOT = Path(os.environ.get("MOPD_ROOT", "/ssd/lzx/mopd"))
TRAIN_PY = ROOT / "mopd_train/bin/python"  # training / grading environment (transformers 5.x)
INFER_PY = ROOT / "mopd_infer/bin/python"  # vLLM environment (transformers 4.55)
SERVE_PY = ROOT / "mopd_serve/bin/python"  # bench service's vLLM (0.31: newer architectures, e.g. Qwen3.5)
MERGEKIT = ROOT / "mopd_merge/bin/mergekit-yaml"
BASE_MODEL = ROOT / "models/Qwen3-4B-Base"
PROFILE_CACHE = ROOT / "cache/profile"
SHM = Path("/dev/shm/mopd")  # per-run shared buffers (published student weights)
GPU_LEASES = Path("/dev/shm/mopd-gpu-leases")  # cross-process GPU claims (runtime.scheduler)
# GPUs reserved for the benchmark service (scripts/bench_service.py): equal workers, one resident vLLM
# each, every job running all its benchmarks on one engine. Training, sweeps, queues and evals never
# get them (runtime.scheduler.usable_gpus filters them out).
SERVICE_GPUS = (6, 7)
# Full-parameter (sharded) training runs on exactly these; GPUs 4-5 stay free for evaluations, which
# otherwise wait hours behind back-to-back trainings that would take every free GPU.
TRAIN_GPUS = (0, 1, 2, 3)
RESERVED_GPUS = frozenset(SERVICE_GPUS)
