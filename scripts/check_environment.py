"""No model downloads: test BF16 attention/backward and NCCL across all ranks."""

import os
import json
import datetime
import torch
import torch.distributed as dist
from transformers import Qwen3Config, Qwen3ForCausalLM

rank = int(os.environ.get("LOCAL_RANK", "0"))
torch.cuda.set_device(rank)
dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))
x = torch.randn(2, 4, 32, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
y = torch.nn.functional.scaled_dot_product_attention(x, x, x, is_causal=True)
y.float().square().mean().backward()
assert torch.isfinite(x.grad).all()
config = Qwen3Config(
    vocab_size=256,
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
)
model = Qwen3ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
ids = torch.randint(0, 256, (2, 32), device="cuda")
loss = model(ids, labels=ids).loss
loss.backward()
assert torch.isfinite(loss)
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
value = torch.tensor([rank + 1.0], device="cuda")
dist.all_reduce(value)
world = dist.get_world_size()
assert value.item() == world * (world + 1) / 2
print(
    json.dumps(
        {
            "rank": rank,
            "gpu": torch.cuda.get_device_name(rank),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "capability": torch.cuda.get_device_capability(rank),
            "qwen3_loss": loss.item(),
            "nccl_sum": value.item(),
            "status": "PASS",
        }
    ),
    flush=True,
)
dist.destroy_process_group()
