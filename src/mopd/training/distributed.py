"""Process-group setup shared by the training entry points (scripts/train_sft.py,
scripts/train_multidomain_opd.py), all launched through torchrun for any GPU count."""
import os

import torch
import torch.distributed as dist


def dist_info():
    """(rank, world_size, device) from the torchrun environment, or (0, 1, None) outside it.
    device is None when not distributed, so the caller falls back to config["student_device"]."""
    world_size=int(os.environ.get("WORLD_SIZE",1))
    if world_size==1:return 0,1,None
    local_rank=int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # On para5090 (8x RTX 5090, no P2P) NCCL's shared-memory transport with the Simple/LL128
    # protocols hits "illegal memory access" once a gradient tensor passes ~1 MB (LoRA on the MLP
    # layers); LL is unaffected and costs ~0.08 s per step for 132M gradient values.
    os.environ.setdefault("NCCL_PROTO", "LL")
    dist.init_process_group(backend="nccl")
    return int(os.environ["RANK"]),world_size,f"cuda:{local_rank}"

