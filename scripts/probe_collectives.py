"""Preflight for full-parameter (FSDP) training: time one model's worth of bf16 all-gather across
the GPUs and exit 0 only if it is fast enough to train.

On para5090 the GPUs have no P2P, so NCCL moves data through host memory; when another job
saturates the host's CPUs and memory bandwidth (2026-10-24: a 64-core PipeANN index build), FSDP
collectives slow ~20-40x and a 4-second step takes ~250 s. Small all-reduces (LoRA-sized) barely
notice, so only a model-sized probe tells.

  torchrun --standalone --nproc_per_node=8 scripts/probe_collectives.py --max-seconds 3
"""
import argparse, os, sys, time
import torch
import torch.distributed as dist

ap = argparse.ArgumentParser()
ap.add_argument("--max-seconds", type=float, default=3.0, help="pass if 36 Qwen3-4B layers gather within this")
a = ap.parse_args()
os.environ.setdefault("NCCL_PROTO", "LL")  # as in training (mopd.training.distributed)
dist.init_process_group("nccl"); rank = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(rank); world = dist.get_world_size()
n = 100930816 // world  # one Qwen3-4B decoder layer
shard = torch.ones(n, dtype=torch.bfloat16, device="cuda"); full = torch.empty(n * world, dtype=torch.bfloat16, device="cuda")
dist.all_gather_into_tensor(full, shard); torch.cuda.synchronize()
start = time.time()
for _ in range(36):
    dist.all_gather_into_tensor(full, shard)
torch.cuda.synchronize()
seconds = torch.tensor(time.time() - start, device="cuda"); dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
ok = float(seconds) <= a.max_seconds
if rank == 0:
    print(f"bf16 all-gather of one model over {world} GPUs: {float(seconds):.2f}s ({'ok' if ok else 'too slow'}, limit {a.max_seconds}s)", flush=True)
dist.destroy_process_group()
sys.exit(0 if ok else 1)
