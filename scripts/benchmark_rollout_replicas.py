"""Measure replica concurrency on identical eight prompts and fixed token work."""
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import LoraConfig,get_peft_model
from train_multidomain_opd import plan
from mopd.rollout.replicas import RolloutReplicas
root=Path("/ssd/lzx/mopd");c=json.loads((root/"configs/dual_domain.json").read_text());c["max_new_tokens"]=256
base=AutoModelForCausalLM.from_pretrained(c["student"],torch_dtype=torch.bfloat16,attn_implementation="sdpa",local_files_only=True).to(c["student_device"])
student=get_peft_model(base,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.,target_modules=["q_proj","k_proj","v_proj","o_proj"],task_type="CAUSAL_LM"))
replicas=RolloutReplicas(c,student);workers=list(replicas.workers);rows=plan(c)[0]
replicas.max_new=8;replicas.generate(student,rows,0,sample=False,min_new_tokens=8);replicas.max_new=256
reports=[]
for n in [1,2,2,1]:
 replicas.workers=workers[:n]
 output,seconds,stats=replicas.generate(student,rows,0,sample=False,min_new_tokens=256)
 result={"workers":n,"seconds":seconds,"tokens":sum(len(r["answer"]) for r in output),"detail":stats}
 result["tokens_per_second"]=result["tokens"]/seconds
 reports.append(result);print(json.dumps(result),flush=True)
(root/"logs/mgpu-rollout-benchmark.json").write_text(json.dumps(reports,indent=2))
