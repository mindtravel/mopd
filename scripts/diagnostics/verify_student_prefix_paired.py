"""Like verify_student_prefix.py, but replays the exact training micro-batches.

rollouts.jsonl is written chunk by chunk, so consecutive same-domain rows of a
step form one micro-batch; packing them identically removes bf16 batch-shape noise.
"""
import argparse,json
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from mopd.data.templates import apply_chat_template_ids,apply_chat_template_text
from mopd.training.dual_template import question
from mopd.training.batching import pack,logps

def teacher_content(row):
    return question(row["prompt"]) if row["domain"]=="math" else row["prompt"]

def load(path,device):
    return AutoModelForCausalLM.from_pretrained(path,dtype=torch.bfloat16,local_files_only=True).to(device).eval()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",required=True);p.add_argument("--student",required=True)
    p.add_argument("--device",default="cuda:0");p.add_argument("--domain",required=True)
    p.add_argument("--micro-batch",type=int,default=2);p.add_argument("--chunks",type=int,default=8)
    p.add_argument("--formats",required=True,help="JSON list; 'chat' means student chat template")
    a=p.parse_args()
    rows=[json.loads(l) for l in open(a.run+"/rollouts.jsonl")]
    rows=[r for r in rows if r["step"]==1 and r["domain"]==a.domain]
    chunks=[rows[i:i+a.micro_batch] for i in range(0,len(rows),a.micro_batch)][:a.chunks]
    ttok=AutoTokenizer.from_pretrained(rows[0]["teacher"],local_files_only=True);teacher=load(rows[0]["teacher"],a.device)
    tlp=[]
    for chunk in chunks:
        prefixes=[apply_chat_template_ids(ttok,[{"role":"user","content":teacher_content(r)}],add_generation_prompt=True,enable_thinking=False) for r in chunk]
        with torch.no_grad():tlp.append([v.float().cpu() for v in logps(teacher,*pack([x+r["response_ids"] for x,r in zip(prefixes,chunk)],list(map(len,prefixes)),ttok.pad_token_id,a.device))])
    del teacher;torch.cuda.empty_cache()
    stok=AutoTokenizer.from_pretrained(a.student,local_files_only=True);student=load(a.student,a.device)
    result={}
    for fmt in json.loads(a.formats):
        render=(lambda r:apply_chat_template_text(stok,[{"role":"user","content":teacher_content(r)}],add_generation_prompt=True,enable_thinking=False)) if fmt=="chat" else (lambda r:fmt.format(teacher_content(r)))
        errs=[]
        for chunk,tv in zip(chunks,tlp):
            prefixes=[stok.encode(render(r)) for r in chunk]
            with torch.no_grad():sv=[v.float().cpu() for v in logps(student,*pack([x+r["response_ids"] for x,r in zip(prefixes,chunk)],list(map(len,prefixes)),stok.pad_token_id,a.device))]
            errs+=[(len(r["response_ids"]),abs(float((s-t).mean())-r["sampled_log_ratio"])) for s,t,r in zip(sv,tv,chunk)]
        result[fmt]={"mean_abs_err":sum(e for _,e in errs)/len(errs),"max_abs_err":max(e for _,e in errs),"rows":errs}
        print(json.dumps({"format":fmt,"mean_abs_err":result[fmt]["mean_abs_err"],"max_abs_err":result[fmt]["max_abs_err"],
            "per_row":[(n,round(e,5)) for n,e in errs]}),flush=True)

if __name__=="__main__":main()
