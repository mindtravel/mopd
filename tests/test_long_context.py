import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "scripts")]
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from mopd.training.long_context import selected_logps
from mopd.training.dual_template import response_logps

torch.manual_seed(23)
m = Qwen3ForCausalLM(
    Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
).eval()
ids = torch.randint(0, 64, (1, 19))
expected = response_logps(m(ids).logits, ids, 4)
expected.sum().backward()
grads = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
m.zero_grad(set_to_none=True)
actual = selected_logps(m, ids, 4, chunk_size=3)
torch.testing.assert_close(actual, expected)
actual.sum().backward()
for n, p in m.named_parameters():
    if n in grads:
        torch.testing.assert_close(p.grad, grads[n], atol=2e-5, rtol=2e-5)
print("PASS: selected log probabilities and all parameter gradients match full logits")
