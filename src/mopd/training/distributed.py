"""Process-group setup shared by the training entry points (scripts/train_sft.py,
scripts/train_multidomain_opd.py), all launched through torchrun for any GPU count."""

import os

import torch
import torch.distributed as dist


def dist_info():
    """(rank, world_size, device) from the torchrun environment, or (0, 1, None) outside it.
    device is None when not distributed, so the caller falls back to config["student_device"]."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size == 1:
        return 0, 1, None
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # NCCL 2.26.2 (torch 2.7.1's pin) hit "illegal memory access" on para5090's shared-memory
    # transport with the Simple/LL128 protocols; 2.28.3 (requirements.lock.txt) fixes it, and its
    # default protocol gathers/reduces 2.4-2.7x faster than the LL-only workaround did.
    dist.init_process_group(backend="nccl")
    return int(os.environ["RANK"]), world_size, f"cuda:{local_rank}"


# Set by torchrun in every rank. A child process that is its own single-process program (a vLLM
# rollout or eval engine) must not inherit them: vLLM reads RANK/WORLD_SIZE/MASTER_* and waits
# forever to join the trainer's process group (full-parameter MOPD smoke, 2026-10-24).
TORCHRUN_VARS = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "GROUP_WORLD_SIZE",
    "ROLE_RANK",
    "ROLE_NAME",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_USE_AGENT_STORE",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
)


def engine_env(**overrides):
    """Environment for a vLLM engine subprocess: no torchrun variables, and the engine core inside
    the process (VLLM_ENABLE_V1_MULTIPROCESSING=0), so killing it leaves no orphaned EngineCore."""
    env = {k: v for k, v in os.environ.items() if k not in TORCHRUN_VARS}
    return {**env, "VLLM_ENABLE_V1_MULTIPROCESSING": "0", **overrides}
