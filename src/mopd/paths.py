"""Every location the project depends on, in one place. MOPD_ROOT (exported by scripts/env.sh)
overrides the project root; everything else derives from it."""

import os
from pathlib import Path

ROOT = Path(os.environ.get("MOPD_ROOT", "/ssd/lzx/mopd"))
TRAIN_PY = ROOT / "mopd_train/bin/python"  # training / grading environment (transformers 5.x)
INFER_PY = ROOT / "mopd_infer/bin/python"  # vLLM environment (transformers 4.55)
MERGEKIT = ROOT / "mopd_merge/bin/mergekit-yaml"
BASE_MODEL = ROOT / "models/Qwen3-4B-Base"
PROFILE_CACHE = ROOT / "cache/profile"
SHM = Path("/dev/shm/mopd")  # per-run shared buffers (published student weights)
GPU_LEASES = Path("/dev/shm/mopd-gpu-leases")  # cross-process GPU claims (runtime.scheduler)
