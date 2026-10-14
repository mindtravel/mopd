"""Fixed held-out trajectories; exact full-vocabulary reverse KL at each response position."""
import json,time,hashlib,re
from pathlib import Path
import torch,pyarrow.parquet as pq
from transformers import AutoTokenizer,AutoModelForCausalLM
from mopd.training.batching import generation_eos_ids
from peft import PeftModel
from mopd.training.dual_template import student_inputs,teacher_inputs
R=Path("/ssd/lzx/mopd");RUN=R/"runs/20261013-numina-8192-batch2-opd";OUT=RUN/"validation"
def readrows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def main():
 OUT.mkdir(exist_ok=True)
 torch.manual_seed(89123)
 tok=AutoTokenizer.from_pretrained(R/"models/Qwen3-4B-Base")
 tt=AutoTokenizer.from_pretrained(R/"models/Qwen3-4B")
 eos_ids=generation_eos_ids(s,tok)
 kw=dict(torch_dtype=torch.bfloat16,attn_implementation="sdpa")
 s=AutoModelForCausalLM.from_pretrained(R/"models/Qwen3-4B-Base",**kw).to("cuda:4").eval()
 t=AutoModelForCausalLM.from_pretrained(R/"models/Qwen3-4B",**kw).to("cuda:5").eval()
 s.requires_grad_(False);t.requires_grad_(False)
 fixed=OUT/"fixed.json"
 if fixed.exists():samples=json.loads(fixed.read_text())
 else:
  norm=lambda x:re.sub(r"[^a-z0-9]","",x.lower())
  excluded={norm(r["problem"]) for r in readrows(R/"data/numina/train.jsonl")}
  for p in (R/"data/math/raw").glob("*/test-*.parquet"):
   excluded.update(norm(r["problem"]) for r in pq.read_table(p,columns=["problem"]).to_pylist())
  samples=[]
  for i,r in enumerate(pq.read_table(R/"data/numina/train-00000.parquet",columns=["problem","source"]).to_pylist()):
   if r["source"] not in ["olympiads","amc_aime"] or norm(r["problem"]) in excluded or not 100<=len(r["problem"])<=1800:continue
   excluded.add(norm(r["problem"]))
   prefix=student_inputs(tok,r["problem"],"cuda:4")
   with torch.no_grad():
    ids=s.generate(input_ids=prefix,attention_mask=torch.ones_like(prefix),do_sample=False,
      max_new_tokens=256,pad_token_id=tok.pad_token_id,eos_token_id=list(eos_ids))
   samples.append({"id":f"numina-shard0:{i}","problem":r["problem"],"student_ids":ids[0].tolist(),"start":prefix.shape[1]})
   if len(samples)==8:break
  fixed.write_text(json.dumps(samples,ensure_ascii=False,indent=2))
  (OUT/"README.md").write_text("Fixed 8 held-out NuminaMath problems; excludes normalized exact overlap with all 1024 training candidates and all MATH test problems. No semantic overlap guarantee. Base greedy responses capped at 256 tokens and frozen. Full-vocabulary reverse KL is exact at these fixed positions, NOT expected KL across fresh policy trajectories or task accuracy. Equal weight per example. Baseline step 0 plus saved checkpoints every 16 steps; this run does not save every 8 steps.")
 digest=hashlib.sha256(fixed.read_bytes()).hexdigest()
 results=OUT/"kl.jsonl"
 done={r["step"] for r in readrows(results)} if results.exists() else set()
 while True:
  for step in [0,16,32,48,64]:
   if step in done:continue
   ck=RUN/f"checkpoint-{step:04d}"
   if step and not (ck/"training_state.pt").exists():continue
   model=PeftModel.from_pretrained(s,ck).eval() if step else s
   values=[]
   with torch.no_grad():
    base=model.get_base_model() if hasattr(model,"get_base_model") else model
    for item in samples:
     ids=torch.tensor([item["student_ids"]],device="cuda:4");start=item["start"]
     ti,ts=teacher_inputs(tt,item["problem"],ids[:,start:],"cuda:5")
     sh=base.model(ids,use_cache=False).last_hidden_state[0,start-1:-1]
     th=t.model(ti,use_cache=False).last_hidden_state[0,ts-1:-1]
     total=0.
     for pos in range(0,len(sh),32):
      sp=base.lm_head(sh[pos:pos+32]).float().log_softmax(-1)
      tp=t.lm_head(th[pos:pos+32]).float().log_softmax(-1).to("cuda:4")
      total+=(sp.exp()*(sp-tp)).sum(-1).sum().item()
     values.append(total/len(sh))
   row={"step":step,"reverse_kl":sum(values)/len(values),"per_example":values,"samples":len(values),"fixed_sha256":digest,"time":time.time()}
   assert all(torch.isfinite(torch.tensor(values))) and min(values)>-1e-5
   with results.open("a") as f:f.write(json.dumps(row)+"\n")
   print(json.dumps(row),flush=True);done.add(step)
   if step:s=model.unload();s.eval()
  if 64 in done:break
  if (RUN/"status.json").exists():
   try:
    if json.loads((RUN/"status.json").read_text()).get("state")=="failed":raise RuntimeError("Training failed")
   except json.JSONDecodeError:pass
  time.sleep(30)
if __name__=="__main__":main()
