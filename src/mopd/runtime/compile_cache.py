"""Per-process compiler cache directories for concurrently starting vLLM engines.

torch.compile / Triton / vLLM caches are written in place; several engines compiling the same new
configuration at once into one shared directory race on its temp-file renames (seen as a
FileNotFoundError in one of 7 sharded-eval engines). Give each concurrent process its own stable
directory: it is still reused by the next run in the same slot.
"""
from pathlib import Path

FOLDERS = (("TORCHINDUCTOR_CACHE_DIR", "torch"), ("TRITON_CACHE_DIR", "triton"), ("VLLM_CACHE_ROOT", "vllm"))


def isolated(root):
    """Env overrides pointing every compiler cache at root/<kind>, created if missing."""
    env = {}
    for key, folder in FOLDERS:
        dest = Path(root) / folder
        dest.mkdir(parents=True, exist_ok=True)
        env[key] = str(dest)
    return env
