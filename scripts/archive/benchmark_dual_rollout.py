"""Controlled fixed-length rollout benchmark; not a quality evaluation."""
import json,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import LoraConfig,get_peft_model
from train_multidomain_opd import plan
from mopd.rollout.hf_batch import generate
root=Path("/ssd/lzx/mopd");c=json.loads((root/"configs/dual_domain.json").read_text())
torch.manual_seed(c["seed"]);tok=AutoTokenizer.from_pretrained(c["student"]);tok.padding_side="left"
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
base=AutoModelForCausalLM.from_pretrained(c["student"],torch_dtype=torch.bfloat16,attn_implementation="sdpa",local_files_only=True).to(c["student_device"])
student=get_peft_model(base,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.,target_modules=["q_proj","k_proj","v_proj","o_proj"],task_type="CAUSAL_LM"))
rows=plan(c)[0]
generate(student,tok,rows,c["student_device"],8,4,sample=False,min_new_tokens=8)
results=[]
for batch in [2,4,4,2]:
 torch.cuda.reset_peak_memory_stats(c["student_device"])
 output,seconds=generate(student,tok,rows,c["student_device"],256,batch,sample=False,min_new_tokens=256)
 result={"generation_batch":batch,"prompts":4,"generated_tokens":sum(len(r["answer"]) for r in output),
   "seconds":seconds,"peak_gib":torch.cuda.max_memory_allocated(c["student_device"])/2**30}
 result["tokens_per_second"]=result["generated_tokens"]/seconds
 results.append(result);print(json.dumps(result),flush=True)
(root/"logs/dual-rollout-benchmark.json").write_text(json.dumps({"note":"256 forced tokens each, greedy, same four prompts, warmed up, no teacher loaded; not representative of long heterogeneous completions","results":results},indent=2))
