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
# What each GPU is for -- the only place GPU ids are written. Jobs ask for a role, not ids
# (runtime.scheduler: "train" = exactly the training GPUs, "eval" = a fair share of the eval GPUs).
#   train    full-parameter / sharded training (4 GPUs), the MOPD student
#   eval     benchmark generation; shares the training GPUs, between trainings
#   service  the checkpoint benchmark service (scripts/bench_service.py); never scheduled
# GPUs 4-5 are left to other users of the server (2026-10-25): no role, never scheduled by us.
GPU_ROLES = {"train": (0, 1, 2, 3), "eval": (0, 1, 2, 3), "service": (6, 7)}
TRAIN_GPUS, EVAL_GPUS, SERVICE_GPUS = GPU_ROLES["train"], GPU_ROLES["eval"], GPU_ROLES["service"]
SCHEDULED_GPUS = tuple(sorted(set(TRAIN_GPUS) | set(EVAL_GPUS)))  # what sweeps, experiments, queues and eval suites schedule
RESERVED_GPUS = frozenset(range(8)) - frozenset(SCHEDULED_GPUS)  # never in a job: the service's and other users'
