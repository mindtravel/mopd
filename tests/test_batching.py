import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "scripts")]
import torch
from types import SimpleNamespace
from transformers import Qwen3Config,Qwen3ForCausalLM
from mopd.training.batching import pack,logps,trim_generated,generation_eos_ids,is_eos_token
from mopd.training.dual_template import response_logps
torch.manual_seed(1)
m=Qwen3ForCausalLM(Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8)).eval()
seqs=[[1,3,4,6,7],[2,5,9,10,11,12,13,14]];starts=[2,3]
reference=[response_logps(m(torch.tensor([s])).logits,torch.tensor([s]),n) for s,n in zip(seqs,starts)]
sum(v.mean() for v in reference).div(2).backward()
grads={n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None}
m.zero_grad(set_to_none=True)
ids,att,mask=pack(seqs,starts,0,"cpu");actual=logps(m,ids,att,mask)
for a,b in zip(actual,reference):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)
sum(v.mean() for v in actual).div(2).backward()
for n,p in m.named_parameters():
 if n in grads:torch.testing.assert_close(p.grad,grads[n],atol=2e-5,rtol=2e-5)
assert trim_generated(torch.tensor([5,6,0,0,0]),0)==[5,6,0]
fake_model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[0,9]),config=SimpleNamespace(eos_token_id=0))
fake_tokenizer=SimpleNamespace(eos_token_id=0)
eos_ids=generation_eos_ids(fake_model,fake_tokenizer)
assert eos_ids==(0,9)
assert trim_generated(torch.tensor([5,6,9,0]),eos_ids)==[5,6,9]
assert is_eos_token(9,eos_ids) and not is_eos_token(8,eos_ids)
print("PASS: padded batch matches single-example logps and sample-mean gradients; EOS preserved")
