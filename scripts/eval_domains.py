"""Fixed held-out math/IFEval evaluation; official strict IFEval checks."""
import argparse,json,os,random,sys,time
from pathlib import Path
ROOT=Path("/ssd/lzx/mopd")
os.environ["NLTK_DATA"]=str(ROOT/"cache/nltk")
sys.path.insert(0,str(ROOT/"vendor"))
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel
from langdetect import DetectorFactory
from instruction_following_eval import evaluation_lib
from mopd.data.domains import student_prompt,teacher_content
from mopd.data.templates import apply_chat_template_ids
from mopd.training.batching import generation_eos_ids,is_eos_token
from eval_math_opd import grade
DetectorFactory.seed=0

def check(row,text):
    if row["domain"]=="math":return grade(row["solution"],text)
    meta=row["metadata"]
    inp=evaluation_lib.InputExample(key=meta["key"],instruction_id_list=meta["instruction_id_list"],prompt=row["prompt"],kwargs=meta["kwargs"])
    result=evaluation_lib.test_instruction_following_strict(inp,{row["prompt"]:text})
    return {"correct":bool(result.follow_all_instructions),"instruction_checks":[bool(v) for v in result.follow_instruction_list]}
def summarize(results):
    summary={}
    for domain in sorted({r["domain"] for r in results}):
        rows=[r for r in results if r["domain"]==domain]
        checks=[x for r in rows for x in r.get("instruction_checks",[])]
        summary[domain]={"count":len(rows),"correct":sum(r["correct"] for r in rows),
           "accuracy":sum(r["correct"] for r in rows)/len(rows),"truncated":sum(r["truncated"] for r in rows),
           "grade_errors":sum(bool(r.get("grade_error")) for r in rows)}
        if checks:summary[domain]["strict_instruction_accuracy"]=sum(checks)/len(checks)
    return summary
def main():
    p=argparse.ArgumentParser()
    for key in ("model","out","data","device"):p.add_argument("--"+key,required=True)
    p.add_argument("--adapter");p.add_argument("--teacher",action="store_true")
    p.add_argument("--max-new-tokens",type=int,default=8192);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    (out/"config.json").write_text(json.dumps(vars(a),indent=2))
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    m=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation="sdpa").to(a.device).eval()
    if a.adapter:m=PeftModel.from_pretrained(m,a.adapter).eval()
    eos_ids=generation_eos_ids(m,tok)
    rows=[]
    for line in Path(a.data).read_text().splitlines():
        if not line.strip(): continue
        row=json.loads(line)
        row.setdefault("prompt",row.get("problem","")); row.setdefault("domain","math")
        rows.append(row)
    results=[];start=time.time()
    with (out/"cases.jsonl").open("w",buffering=1) as f:
        for row in rows:
            if a.teacher:
                prompt=apply_chat_template_ids(tok,[{"role":"user","content":teacher_content(row)}],add_generation_prompt=True,enable_thinking=False,return_tensors="pt").to(a.device)
            else:prompt=tok(student_prompt(tok,row),return_tensors="pt").input_ids.to(a.device)
            with torch.no_grad():
                ids=m.generate(input_ids=prompt,attention_mask=torch.ones_like(prompt),do_sample=False,temperature=None,top_p=None,top_k=None,
                   max_new_tokens=a.max_new_tokens,eos_token_id=list(eos_ids),pad_token_id=tok.pad_token_id or tok.eos_token_id)
            answer=ids[0,prompt.shape[1]:];text=tok.decode(answer,skip_special_tokens=True)
            score=check(row,text)
            result={"id":row["id"],"domain":row["domain"],"prompt":row["prompt"],"response":text,
              "tokens":len(answer),"truncated":not is_eos_token(answer[-1],eos_ids),**score}
            results.append(result);f.write(json.dumps(result,ensure_ascii=False)+"\n")
            report={"done":len(results),"total":len(rows),"seconds":time.time()-start,"domains":summarize(results)}
            (out/"progress.json").write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    report["scope"]="Fixed pilot heldout subset; not full IFEval or full MATH benchmark."
    (out/"summary.json").write_text(json.dumps(report,indent=2))
if __name__=="__main__":main()
