import argparse,json,re
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel
from mopd.training.batching import generation_eos_ids,trim_generated
PROMPTS=[]
for a,b in [(17,23),(19,21),(24,18),(31,14),(27,16),(42,13),(36,25),(48,17),(55,19),(63,28),(71,12),(84,16),(93,27),(105,18),(117,23),(128,14),(139,31),(146,22),(157,13),(168,21)]:
 PROMPTS.append((f"Calculate {a} * {b}. Give only the number.",str(a*b)))
def load(path,device):
 tok=AutoTokenizer.from_pretrained(path); m=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.bfloat16,attn_implementation="sdpa").to(device).eval(); return tok,m
def run(tok,m,prompt,device):
 text=f"Question: {prompt}\nAnswer:"; x=tok(text,return_tensors="pt").to(device)
 eos_ids=generation_eos_ids(m,tok)
 with torch.no_grad(): y=m.generate(**x,do_sample=False,max_new_tokens=16,pad_token_id=tok.pad_token_id,eos_token_id=list(eos_ids))
 return tok.decode(trim_generated(y[0,x.input_ids.shape[1]:],eos_ids),skip_special_tokens=True).strip()
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--base",required=True);ap.add_argument("--teacher",required=True);ap.add_argument("--adapter");ap.add_argument("--out",required=True);a=ap.parse_args()
 device="cuda:0"; tok,s=load(a.base,device); _,t=load(a.teacher,"cuda:1")
 models=[("student_before",s,tok,device),("teacher",t,tok,"cuda:1")]
 if a.adapter:
  ft=PeftModel.from_pretrained(s,a.adapter).eval(); models.append(("student_after",ft,tok,device))
 rows=[]
 for name,m,tk,d in models:
  good=0
  for p,ans in PROMPTS:
   out=run(tk,m,p,d); norm=re.sub(r"[^0-9-]","",out.split("\\n")[0]); ok=norm==ans; good+=ok; rows.append({"model":name,"prompt":p,"expected":ans,"output":out,"correct":ok})
  print(name,good,"/",len(PROMPTS),good/len(PROMPTS),flush=True)
 Path(a.out).write_text(json.dumps({"n":len(PROMPTS),"rows":rows},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
