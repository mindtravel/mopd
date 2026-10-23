"""GPU check for mopd.training.fsdp.gather: the streaming bf16 gather must equal casting the
torch.distributed.checkpoint full state dict to bf16, tensor for tensor, on a sharded Qwen3 student.

  torchrun --standalone --nproc_per_node=4 scripts/test_fsdp_gather.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from transformers import AutoModelForCausalLM
from mopd.training import fsdp
from mopd.training.distributed import dist_info

rank, world, device = dist_info()
model = AutoModelForCausalLM.from_pretrained("/ssd/lzx/mopd/models/Qwen3-4B-Base", dtype=torch.float32,
                                             local_files_only=True).to(device)
scorer = fsdp.shard(model)
with torch.no_grad():  # make every shard differ from the base so a mis-assembled tensor would show
    for p in scorer.parameters():
        fsdp.local(p).add_(torch.randn_like(fsdp.local(p)) * 1e-3)
fast = fsdp.gather(scorer, rank == 0)
ref = get_model_state_dict(scorer, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
if rank == 0:
    ref = {k.removeprefix("base."): v.to(torch.bfloat16) for k, v in ref.items()}
    missing = set(ref) - set(fast)
    assert missing <= {"lm_head.weight"}, missing  # tied: the embedding, gathered once
    bad = [k for k in fast if not torch.equal(fast[k], ref[k])]
    assert not bad, bad[:5]
    print(f"gather ok: {len(fast)} tensors identical to the reference (tied lm_head omitted: {bool(missing)})")
dist.barrier(); dist.destroy_process_group()
