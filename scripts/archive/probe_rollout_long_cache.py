"""Capacity probe: full-size synthetic KV cache plus real decode; not quality test."""
import gc,json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,DynamicCache
def main():
 root=Path("/ssd/lzx/mopd");torch.set_num_threads(2);device="cuda:7"
 m=AutoModelForCausalLM.from_pretrained(root/"models/Qwen3-4B-Base",local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation="sdpa").to(device).eval()
 length=json.loads((root/"logs/dual-prompt-lengths.json").read_text())["max_prompt_tokens"]+8192
 reports=[]
 for batch in [8,16]:
  cache=DynamicCache()
  torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
  try:
   for layer in range(m.config.num_hidden_layers):
    shape=(batch,m.config.num_key_value_heads,length,m.config.head_dim)
    cache.update(torch.zeros(shape,dtype=torch.bfloat16,device=device),torch.zeros(shape,dtype=torch.bfloat16,device=device),layer)
   with torch.no_grad():
    output=m(input_ids=torch.ones((batch,1),dtype=torch.long,device=device),
      attention_mask=torch.ones((batch,length+1),dtype=torch.long,device=device),
      past_key_values=cache,cache_position=torch.tensor([length],device=device),
      position_ids=torch.full((batch,1),length,device=device),use_cache=True,logits_to_keep=1)
   torch.cuda.synchronize(device)
   result={"batch":batch,"cached_tokens_per_sequence":length,"peak_gib":torch.cuda.max_memory_allocated(device)/2**30,"finite":bool(torch.isfinite(output.logits).all()),"passed":True}
   del output
  except torch.OutOfMemoryError as e:
   result={"batch":batch,"cached_tokens_per_sequence":length,"passed":False,"error":"CUDA out of memory"}
  reports.append(result);print(json.dumps(result),flush=True)
  del cache;gc.collect();torch.cuda.empty_cache()
 (root/"logs/rollout-long-cache-probe.json").write_text(json.dumps({"scope":"Synthetic zero KV cache plus one real decode token. Tests decode capacity at longest selected prefix + 8192 output; does not replace real rollout validation.","results":reports},indent=2))
if __name__=="__main__":main()
