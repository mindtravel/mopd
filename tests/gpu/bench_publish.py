"""Where full-weight publishing time goes (mopd.rollout.policy -> fsdp.write_weights), on a sharded
Qwen3-4B student:  torchrun --standalone --nproc_per_node=4 tests/gpu/bench_publish.py"""
import sys, time
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_ROOT / "src")]
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM
from mopd.training import fsdp
from mopd.training.distributed import dist_info

rank, world, device = dist_info()
model = AutoModelForCausalLM.from_pretrained("/ssd/lzx/mopd/models/Qwen3-4B-Base", dtype=torch.float32, local_files_only=True).to(device)
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


for _ in range(2):
    timed("gather per parameter, GPU only", per_param_gpu)
    state = timed("gather per parameter + .cpu() (current fsdp.gather)", lambda: fsdp.gather(scorer, main))
    if main:
        from safetensors.torch import save_file
        t = time.perf_counter(); save_file({k: v.contiguous() for k, v in state.items()}, "/dev/shm/bench-publish.safetensors")
        print(f"save_file 8 GB to /dev/shm: {time.perf_counter() - t:.2f}s", flush=True)
        n = sum(v.numel() for v in state.values())
        pinned = torch.empty(n, dtype=torch.bfloat16, pin_memory=True)
        t = time.perf_counter(); off = 0
        for v in state.values():
            pinned[off:off + v.numel()].copy_(v.reshape(-1)); off += v.numel()
        print(f"memcpy into pinned buffer: {time.perf_counter() - t:.2f}s", flush=True)
    dist.barrier()
if main:
    Path("/dev/shm/bench-publish.safetensors").unlink(missing_ok=True)
dist.destroy_process_group()
