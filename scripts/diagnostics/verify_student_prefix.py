"""Recover the student prefix used by a finished run from its step-1 rollouts.

At step 1 a fresh LoRA (B=0) equals the base model, so for the true prefix
mean(student_logp - teacher_logp) reproduces the recorded sampled_log_ratio.
"""
import argparse,json
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from mopd.data.templates import apply_chat_template_ids,apply_chat_template_text
from mopd.training.dual_template import question
from mopd.training.batching import pack,logps

def teacher_content(row):
    return question(row["prompt"]) if row["domain"]=="math" else row["prompt"]

FORMATS=["Question: {}\nAnswer:","Instruction: {}\nAnswer:","Instruction: {}\nResponse:",
    "Instruction: {}\nAnswer: ","Instruction:\n{}\nAnswer:","Instruction: {}\n\nAnswer:","Instruction:\n{}\n\nAnswer:",
    "{}\nAnswer:","{}\n","User: {}\nAssistant:","### Instruction:\n{}\n\n### Response:\n",
    "Instruction: {}\nOutput:","Question: {}\n\nAnswer:","Instruction: {}\nAnswer:\n"]
CANDIDATES={
    **{f:(lambda f:lambda tok,r:f.format(teacher_content(r)))(f) for f in FORMATS},
    "chat_template":lambda tok,r:apply_chat_template_text(tok,[{"role":"user","content":teacher_content(r)}],add_generation_prompt=True,enable_thinking=False),
}

def load(path,device):
    return AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.bfloat16,local_files_only=True).to(device).eval()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",required=True);p.add_argument("--student",required=True)
    p.add_argument("--device",default="cuda:3");p.add_argument("--per-domain",type=int,default=6)
    p.add_argument("--per-row",action="store_true");p.add_argument("--domain")
    p.add_argument("--min-tokens",type=int,default=0);p.add_argument("--formats",help="JSON list overriding FORMATS")
    a=p.parse_args()
    rows=[json.loads(l) for l in open(a.run+"/rollouts.jsonl")]
    rows=[r for r in rows if r["step"]==1 and (a.domain is None or r["domain"]==a.domain) and len(r["response_ids"])>=a.min_tokens]
    candidates=CANDIDATES if not a.formats else {f:(lambda f:lambda tok,r:f.format(teacher_content(r)))(f) for f in json.loads(a.formats)}
    picked=[]
    for d in sorted({r["domain"] for r in rows}):
        picked+=sorted([r for r in rows if r["domain"]==d],key=lambda r:len(r["response_ids"]))[:a.per_domain]
    stok=AutoTokenizer.from_pretrained(a.student,local_files_only=True)
    student=load(a.student,a.device)
    teacher_lp={}
    for path in sorted({r["teacher"] for r in picked}):
        ttok=AutoTokenizer.from_pretrained(path,local_files_only=True);teacher=load(path,a.device)
        for i,r in enumerate(picked):
            if r["teacher"]!=path:continue
            prefix=apply_chat_template_ids(ttok,[{"role":"user","content":teacher_content(r)}],add_generation_prompt=True,enable_thinking=False)
            with torch.no_grad():teacher_lp[i]=logps(teacher,*pack([prefix+r["response_ids"]],[len(prefix)],ttok.pad_token_id,a.device))[0].float().cpu()
        del teacher;torch.cuda.empty_cache()
    out={}
    for name,fn in candidates.items():
        errs={}
        for i,r in enumerate(picked):
            prefix=stok.encode(fn(stok,r))
            with torch.no_grad():sp=logps(student,*pack([prefix+r["response_ids"]],[len(prefix)],stok.pad_token_id,a.device))[0].float().cpu()
            ratio=float((sp-teacher_lp[i]).mean())
            errs.setdefault(r["domain"],[]).append(abs(ratio-r["sampled_log_ratio"]))
            if a.per_row:print("  ",repr(name),r["domain"],r["id"],len(r["response_ids"]),round(ratio,4),round(r["sampled_log_ratio"],4),flush=True)
        out[name]={d:{"mean_abs_err":sum(v)/len(v),"max_abs_err":max(v),"n":len(v)} for d,v in errs.items()}
        print(name,json.dumps(out[name]),flush=True)
    print(json.dumps({"run":a.run,"result":out}))

if __name__=="__main__":main()
