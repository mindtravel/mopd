"""FSDP-sized collectives under NCCL settings given in the environment: one model's bf16 all-gather
and fp32 reduce-scatter (36 Qwen3-4B layers), timed, plus a crash check (the Simple/LL128 protocols
once hit "illegal memory access" on this machine's shared-memory transport).

  NCCL_PROTO=Simple NCCL_SHM_USE_CUDA_MEMCPY=1 torchrun --standalone --nproc_per_node=4 tests/gpu/bench_nccl.py
"""
import os, time
import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(rank); world = dist.get_world_size()
n = 100930816 // world
out = {}
for name, dtype, op in (("all_gather_bf16", torch.bfloat16, "ag"), ("reduce_scatter_fp32", torch.float32, "rs")):
    shard = torch.ones(n, dtype=dtype, device="cuda"); full = torch.ones(n * world, dtype=dtype, device="cuda")
    run = (lambda: dist.all_gather_into_tensor(full, shard)) if op == "ag" else (lambda: dist.reduce_scatter_tensor(shard, full))
    run(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(36):
        run()
    torch.cuda.synchronize()
    sec = torch.tensor(time.perf_counter() - t, device="cuda"); dist.all_reduce(sec, op=dist.ReduceOp.MAX)
    out[name] = round(float(sec), 2)
if rank == 0:
    keys = ("NCCL_PROTO", "NCCL_SHM_USE_CUDA_MEMCPY", "NCCL_SHM_DISABLE", "NCCL_P2P_DISABLE", "NCCL_ALGO")
    print({**{k: os.environ.get(k) for k in keys if os.environ.get(k)}, **out}, flush=True)
dist.destroy_process_group()
