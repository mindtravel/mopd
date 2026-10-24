"""Batched full benchmark evaluation; generation batching, per-row scoring."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path("/ssd/lzx/mopd");os.environ["NLTK_DATA"]=str(ROOT/"cache/nltk");sys.path.insert(0,str(ROOT/"vendor"))
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel
from langdetect import DetectorFactory
from instruction_following_eval import evaluation_lib
from mopd.data.domains import student_prompt,teacher_content
from mopd.data.templates import apply_chat_template_ids
from eval_math_opd import grade
DetectorFactory.seed=0
from eval_domains import check as score, summarize
from mopd.training.batching import trim_generated,generation_eos_ids,is_eos_token
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--model",required=True);p.add_argument("--data",required=True);p.add_argument("--out",required=True);p.add_argument("--device",required=True);p.add_argument("--adapter");p.add_argument("--teacher",action="store_true");p.add_argument("--max-new-tokens",type=int,default=8192);p.add_argument("--batch",type=int,default=4)
 a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 (out/"config.json").write_text(json.dumps(vars(a),indent=2))
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);tok.padding_side="left"
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation="sdpa").to(a.device).eval()
 if a.adapter:model=PeftModel.from_pretrained(model,a.adapter).eval()
 eos_ids=generation_eos_ids(model,tok)
 rows=[]
 for line in Path(a.data).read_text().splitlines():
  if line.strip():
   r=json.loads(line);r.setdefault("prompt",r.get("problem",""));r.setdefault("domain","math");rows.append(r)
 results=[];started=time.time()
 with (out/"cases.jsonl").open("w",buffering=1) as f:
  for off in range(0,len(rows),a.batch):
   batch=rows[off:off+a.batch]
   texts=[teacher_content(r) if a.teacher else student_prompt(tok,r) for r in batch]
   if a.teacher:
    ids=[apply_chat_template_ids(tok,[{"role":"user","content":x}],add_generation_prompt=True,enable_thinking=False) for x in texts]
    width=max(map(len,ids)); inp=torch.full((len(ids),width),tok.pad_token_id,dtype=torch.long,device=a.device);mask=torch.zeros_like(inp)
    for i,v in enumerate(ids):inp[i,-len(v):]=torch.tensor(v,device=a.device);mask[i,-len(v):]=1
   else:
    x=tok(texts,padding=True,return_tensors="pt");inp=x.input_ids.to(a.device);mask=x.attention_mask.to(a.device)
   with torch.no_grad():gen=model.generate(input_ids=inp,attention_mask=mask,do_sample=False,max_new_tokens=a.max_new_tokens,eos_token_id=list(eos_ids),pad_token_id=tok.pad_token_id,use_cache=True)
   for i,r in enumerate(batch):
    ans=trim_generated(gen[i,inp.shape[1]:],eos_ids);text=tok.decode(ans,skip_special_tokens=True);z=score(r,text)
    item={"id":r["id"],"domain":r["domain"],"prompt":r["prompt"],"response":text,"tokens":len(ans),"truncated":not is_eos_token(ans[-1],eos_ids),**z};results.append(item);f.write(json.dumps(item,ensure_ascii=False)+"\n")
   summary={"done":len(results),"total":len(rows),"seconds":time.time()-started,"max_new_tokens":a.max_new_tokens}
   (out/"progress.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
 summary.update({"domains":summarize(results),"scope":"All rows from the recorded input file; greedy; batched generation","input_file":str(Path(a.data).resolve())})
 (out/"summary.json").write_text(json.dumps(summary,indent=2))
if __name__=="__main__":main()
