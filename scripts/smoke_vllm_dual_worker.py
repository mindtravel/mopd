"""GPU smoke for real LoRA swapping, with GPU HF next-token reference."""
import argparse,json,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
from mopd.rollout.vllm_processes import VllmRollouts
from mopd.data.domains import student_prompt
def main():
 p=argparse.ArgumentParser();p.add_argument("--device",default="cuda:3");p.add_argument("--memory",type=float,default=.45);p.add_argument("--out",required=True);a=p.parse_args()
 root=Path("/ssd/lzx/mopd");out=Path(a.out).resolve()
 if not out.is_relative_to(root):raise ValueError("Output outside project")
 out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(8)
 modelpath=str(root/"models/Qwen3-4B-Base")
 tok=AutoTokenizer.from_pretrained(modelpath,local_files_only=True)
 base=AutoModelForCausalLM.from_pretrained(modelpath,torch_dtype=torch.bfloat16,attn_implementation="sdpa",local_files_only=True)
 model=PeftModel.from_pretrained(base,root/"runs/20261014-dual-domain-batch32-mgpu-opd-b8/checkpoint-0064").eval()
 plan=json.loads((root/"runs/20261014-dual-domain-batch32-mgpu-opd-b8/batch_plan.json").read_text())
 rows=[next(r for r in plan[0] if r["domain"]==d) for d in ("math","instruction")]
 cfg={"student":modelpath,"output":str(out),"rollout_devices":["cuda:6","cuda:7"],"generation_batch_size":1,
   "vllm_memory_utilization":a.memory,"max_new_tokens":16,"seed":20261013,"debug_logprobs":10}
 originals={n:v.detach().cpu().clone() for n,v in model.named_parameters() if "lora_B" in n}
 references=[]
 model.to(a.device)
 for version in (0,1):
  if version:
   with torch.no_grad():
    for n,v in model.named_parameters():
     if "lora_B" in n:v.zero_()
  reference=[]
  for row in rows:
   ids=tok(student_prompt(tok,row),return_tensors="pt").input_ids.to(a.device)
   with torch.no_grad():logits=model(input_ids=ids,use_cache=False,logits_to_keep=1).logits[0,-1].float()
   logp=logits.log_softmax(-1);best=logp.topk(10)
   reference.append({"first_token":best.indices[0].item(),"top_tied":[int(x) for x,y in zip(best.indices.tolist(),best.values.tolist()) if float(best.values[0]-y)<1e-5],"margin":(best.values[0]-best.values[1]).item(),
      "top_logprobs":{str(t):v for t,v in zip(best.indices.tolist(),best.values.tolist())}})
  references.append(reference)
 with torch.no_grad():
  for n,v in model.named_parameters():
   if n in originals:v.copy_(originals[n].to(a.device))
 model.to("cpu")
 del ids,logits,logp,best
 torch.cuda.empty_cache()
 (out/"gpu-reference.json").write_text(json.dumps(references,indent=2))
 backend=VllmRollouts(cfg,model);reports=[]
 try:
  for version in (0,1):
   if version:
    with torch.no_grad():
     for n,v in model.named_parameters():
      if "lora_B" in n:v.zero_()
   reference=references[version]
   result,seconds,stats=backend.generate(model,rows,version,sample=False)
   assert [x["index"] for x in result]==list(range(len(rows))), "Result order mismatch"
   assert {x["rollout_device"] for x in result}=={"cuda:6","cuda:7"}, "Both workers must participate"
   assert all(x["version"]==version for x in stats["workers"]), "Stale worker version"
   assert len({x["adapter_sha256"] for x in stats["workers"]})==1, "Worker weights differ"
   records=[{"id":r["row"]["id"],"prefix":r["prefix"],"answer":r["answer"],
    "reference":ref,"vllm_top_logprobs":r.get("first_logprobs"),"first_token_match":r["answer"][0] in ref["top_tied"]} for r,ref in zip(result,reference)]
   report={"version":version,"seconds":seconds,"stats":stats,"records":records}
   reports.append(report);(out/"results.json").write_text(json.dumps(reports,indent=2))

  if not all(row["first_token_match"] for report in reports for row in report["records"]):raise AssertionError("GPU HF/vLLM next-token mismatch outside HF tied top tokens: inspect candidate logprobs")
  assert reports[0]["stats"]["workers"][0]["adapter_sha256"]!=reports[1]["stats"]["workers"][0]["adapter_sha256"]
  (out/"summary.json").write_text(json.dumps({"state":"passed","versions":2,"checks":"real LoRA replacement; exact prefixes; first greedy token vs GPU HF","limitation":"16 output tokens, not long-output throughput"},indent=2))
 finally:backend.close()
if __name__=="__main__":main()
