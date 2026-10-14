"""HF real request comparator. Same Base, prompts, greedy decoding and EOS as vLLM probe."""
import argparse,json,os,time
from pathlib import Path
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--device",type=int,required=True);p.add_argument("--batch-index",type=int,default=5)
 p.add_argument("--batch",type=int,default=16);p.add_argument("--out",required=True)
 a=p.parse_args()
 if a.device not in (2,3):raise ValueError("Probe restricted to GPUs 2/3")
 os.environ["CUDA_VISIBLE_DEVICES"]=str(a.device)
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM
 from mopd.rollout.hf_batch import generate
 torch.set_num_threads(2)
 root=Path("/ssd/lzx/mopd")
 rows=json.loads((root/"runs/20261013-dual-domain-batch32-mgpu-opd/batch_plan.json").read_text())[a.batch_index]
 out=Path(a.out).resolve()
 if not out.is_relative_to(root):raise ValueError("Output outside project")
 out.mkdir(parents=True,exist_ok=False)
 modelpath=str(root/"models/Qwen3-4B-Base")
 tok=AutoTokenizer.from_pretrained(modelpath,local_files_only=True);tok.padding_side="left";tok.pad_token=tok.eos_token
 model=AutoModelForCausalLM.from_pretrained(modelpath,torch_dtype=torch.bfloat16,attn_implementation="sdpa",local_files_only=True).to("cuda:0")
 generate(model,tok,rows[:1],"cuda:0",8,1,sample=False)
 torch.cuda.reset_peak_memory_stats()
 start=time.perf_counter();batches=[]
 try:
  output=[];seconds=0.
  for offset in range(0,len(rows),a.batch):
   chunk=rows[offset:offset+a.batch]
   torch.cuda.reset_peak_memory_stats()
   items,elapsed=generate(model,tok,chunk,"cuda:0",8192,a.batch,sample=False)
   output.extend(items);seconds+=elapsed
   batches.append({"offset":offset,"count":len(chunk),"seconds":elapsed,
    "longest":max(len(r["answer"]) for r in items),
    "reached_8192":sum(len(r["answer"])==8192 for r in items),
    "peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
    "peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30})
  records=[{"id":r["row"]["id"],"domain":r["row"]["domain"],"prompt":r["row"]["prompt"],
   "response":tok.decode(r["answer"]),"tokens":len(r["answer"]),
   "finish_reason":"stop" if r["answer"][-1]==tok.eos_token_id else "length"} for r in output]
  with (out/"cases.jsonl").open("w") as f:
   for r in records:f.write(json.dumps(r,ensure_ascii=False)+"\n")
  result={"state":"complete","seconds":seconds,"tokens":sum(r["tokens"] for r in records),
   "longest":max(r["tokens"] for r in records),"truncated":sum(r["finish_reason"]=="length" for r in records),"count":len(records)}
  result["tokens_per_second"]=result["tokens"]/seconds
 except torch.OutOfMemoryError:
  result={"state":"oom","seconds":time.perf_counter()-start}
 result["batches"]=batches
 result.update(backend="hf",device=a.device,batch=a.batch,batch_index=a.batch_index,max_new_tokens=8192,
  peak_allocated_gib=max([torch.cuda.max_memory_allocated()/2**30]+[b["peak_allocated_gib"] for b in batches]),
  peak_reserved_gib=max([torch.cuda.max_memory_reserved()/2**30]+[b["peak_reserved_gib"] for b in batches]),
  sampling="greedy; natural EOS; warmup excluded; unadapted Base")
 (out/"result.json").write_text(json.dumps(result,indent=2))
 print(json.dumps(result),flush=True)
if __name__=="__main__":main()
