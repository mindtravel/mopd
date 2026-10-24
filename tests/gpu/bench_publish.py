"""Where full-weight publishing time goes (mopd.rollout.policy -> fsdp.SharedWeights), on a sharded
Qwen3-4B student:  torchrun --standalone --nproc_per_node=4 tests/gpu/bench_publish.py"""
import sys, time
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_ROOT / "src")]
from mopd import paths
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM
from mopd.training import fsdp
from mopd.training.distributed import dist_info

rank, world, device = dist_info()
model = AutoModelForCausalLM.from_pretrained(str(paths.BASE_MODEL), dtype=torch.float32, local_files_only=True).to(device)
scorer = fsdp.shard(model)
main = rank == 0


def timed(name, fn):
    torch.cuda.synchronize(); dist.barrier(); t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize(); dist.barrier()
    if main:
        print(f"{name}: {time.perf_counter() - t:.2f}s", flush=True)
    return out


def per_param_gpu():  # all-gathers only, results stay on GPU and are dropped
    for _, p in scorer.named_parameters():
        f = p.detach().to(torch.bfloat16)
        f = f.full_tensor() if hasattr(f, "full_tensor") else f
        del f


shared = fsdp.SharedWeights(scorer, "/dev/shm/bench-publish.bf16", main)
for _ in range(3):
    timed("gather per parameter, GPU only", per_param_gpu)
    timed("SharedWeights.write (gather + DMA into page-locked shared buffer)", lambda: shared.write(scorer))
    if main:  # what a rollout engine does: map the buffer and copy each tensor to its GPU
        t = time.perf_counter()
        for name, v in fsdp.read_shared_weights(shared.path):
            v.to(device, non_blocking=False)
        torch.cuda.synchronize()
        print(f"map + host-to-device copy of every tensor (rollout side): {time.perf_counter() - t:.2f}s", flush=True)
    dist.barrier()
shared.close()
if main:
    Path("/dev/shm/bench-publish.bf16").unlink(missing_ok=True)
    Path("/dev/shm/bench-publish.bf16.json").unlink(missing_ok=True)
dist.destroy_process_group()
