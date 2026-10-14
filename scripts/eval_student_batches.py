"""Batched student-only pilot evaluation, with the same 8192-token limit."""
import argparse,json,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel
from eval_domains import check,summarize
from mopd.rollout.hf_batch import generate
def main():
 p=argparse.ArgumentParser()
 for k in ("model","out","data","device"):p.add_argument("--"+k,required=True)
 p.add_argument("--adapter");p.add_argument("--batch",type=int,default=4);a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=False);(out/"config.json").write_text(json.dumps(vars(a),indent=2))
 torch.set_num_threads(2)
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);tok.padding_side="left"
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 m=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation="sdpa").to(a.device).eval()
 if a.adapter:m=PeftModel.from_pretrained(m,a.adapter).eval()
 rows=[json.loads(l) for l in Path(a.data).read_text().splitlines() if l.strip()]
 results=[];start=time.time()
 with (out/"cases.jsonl").open("w",buffering=1) as f:
  for offset in range(0,len(rows),a.batch):
   answers,_=generate(m,tok,rows[offset:offset+a.batch],a.device,8192,a.batch,sample=False)
   for item in answers:
    row=item["row"];text=tok.decode(item["answer"],skip_special_tokens=True)
    result={"id":row["id"],"domain":row["domain"],"prompt":row["prompt"],"response":text,
      "tokens":len(item["answer"]),"truncated":item["answer"][-1]!=tok.eos_token_id,**check(row,text)}
    results.append(result);f.write(json.dumps(result,ensure_ascii=False)+"\n")
   report={"done":len(results),"total":len(rows),"seconds":time.time()-start,"domains":summarize(results),"max_new_tokens":8192}
   part=out/"progress.partial";part.write_text(json.dumps(report,indent=2));part.replace(out/"progress.json")
   print(json.dumps(report),flush=True)
 report["scope"]="Fixed 32 MATH + 32 IFEval pilot questions; greedy, identical student template and 8192-token cap; not full benchmarks."
 (out/"summary.json").write_text(json.dumps(report,indent=2))
if __name__=="__main__":main()
