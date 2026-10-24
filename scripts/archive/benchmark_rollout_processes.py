"""Throughput sweep of process workers; fixed 256-token outputs."""
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig,get_peft_model
from train_multidomain_opd import plan
from mopd.rollout.processes import ProcessRollouts
def main():
 root=Path("/ssd/lzx/mopd");c=json.loads((root/"configs/dual_domain.json").read_text())
 c["max_new_tokens"]=256;c["output"]=str(root/"runs/rollout-process-benchmark")
 torch.set_num_threads(2)
 base=AutoModelForCausalLM.from_pretrained(c["student"],torch_dtype=torch.bfloat16,attn_implementation="sdpa",local_files_only=True).to(c["student_device"])
 student=get_peft_model(base,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.,target_modules=["q_proj","k_proj","v_proj","o_proj"],task_type="CAUSAL_LM"))
 replicas=ProcessRollouts(c,student);workers=list(replicas.workers)
 rows=[r for b in plan(c)[:4] for r in b];reports=[]
 try:
  c["max_new_tokens"]=8;replicas.generate(student,rows[:8],0,sample=False,min_new_tokens=8);c["max_new_tokens"]=256
  for n,batch in [(1,4),(2,4),(2,8),(2,16)]:
   replicas.workers=workers[:n];c["generation_batch_size"]=batch
   output,seconds,stats=replicas.generate(student,rows,0,sample=False,min_new_tokens=256)
   result={"workers":n,"batch_per_worker":batch,"prompts":len(rows),"seconds":seconds,
     "tokens":sum(len(r["answer"]) for r in output),"detail":stats}
   result["tokens_per_second"]=result["tokens"]/seconds
   reports.append(result);print(json.dumps(result),flush=True)
   (root/"logs/process-rollout-benchmark.json").write_text(json.dumps(reports,indent=2))
 finally:
  replicas.workers=workers;replicas.close()
if __name__=="__main__":main()
