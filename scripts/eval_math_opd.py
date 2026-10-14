"""Fixed MATH subset evaluation. No adapter is attached to the baseline model."""
import argparse,json,time,traceback
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel
from math_verify import parse,verify
from mopd.training.dual_template import student_inputs,question
from mopd.data.templates import apply_chat_template_ids
from mopd.training.batching import generation_eos_ids,trim_generated

def boxed(text):
    # Require an explicit final answer; never compare arbitrary numbers in a derivation.
    marker="\\boxed{"
    start=text.rfind(marker)
    if start<0:return None
    begin=start+len(marker);depth=1
    for i in range(begin,len(text)):
        if text[i]=="{":depth+=1
        elif text[i]=="}":depth-=1
        if depth==0:return text[begin:i]
    return None

def grade(solution,response):
    gold=boxed(solution);pred=boxed(response)
    if gold is None:return {"correct":False,"grade_error":"missing_gold_box"}
    if pred is None:return {"correct":False,"grade_error":None,"prediction":None}
    try:
        g=parse("$\\boxed{"+gold+"}$");p=parse("$\\boxed{"+pred+"}$")
        return {"correct":bool(verify(g,p)),"grade_error":None,"prediction":pred,"gold":gold}
    except Exception as e:return {"correct":False,"grade_error":repr(e),"prediction":pred,"gold":gold}

def main():
    ap=argparse.ArgumentParser()
    for arg in ["model","out","data","device"]:ap.add_argument("--"+arg,required=True)
    ap.add_argument("--teacher",action="store_true");ap.add_argument("--adapter");ap.add_argument("--max-new-tokens",type=int,default=1024)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    (out/"config.json").write_text(json.dumps(vars(a),indent=2))
    tok=AutoTokenizer.from_pretrained(a.model)
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").to(a.device).eval()
    if a.adapter:model=PeftModel.from_pretrained(model,a.adapter).eval()
    eos_ids=generation_eos_ids(model,tok)
    rows=[json.loads(s) for s in Path(a.data).read_text().splitlines()]
    scores=[];started=time.time()
    with (out/"cases.jsonl").open("w",buffering=1) as f:
        for i,row in enumerate(rows):
            if a.teacher:
                prefix=apply_chat_template_ids(tok,[{"role":"user","content":question(row["problem"])}],
                    add_generation_prompt=True,enable_thinking=False,return_tensors="pt").to(a.device)
            else:
                prefix=student_inputs(tok,row["problem"],a.device)
            with torch.no_grad():
                ids=model.generate(input_ids=prefix,attention_mask=torch.ones_like(prefix),
                    do_sample=False,temperature=None,top_k=None,top_p=None,
                    max_new_tokens=a.max_new_tokens,eos_token_id=list(eos_ids),pad_token_id=tok.pad_token_id)
            answer=trim_generated(ids[0,prefix.shape[1]:],eos_ids);text=tok.decode(answer,skip_special_tokens=True)
            score=grade(row["solution"],text);scores.append(score)
            f.write(json.dumps({"id":row["id"],"problem":row["problem"],"solution":row["solution"],
                "output":text,"tokens":len(answer),"hit_token_limit":len(answer)==a.max_new_tokens,**score},ensure_ascii=False)+"\n")
            result={"done":i+1,"total":len(rows),"correct":sum(s["correct"] for s in scores),
                "accuracy":sum(s["correct"] for s in scores)/len(scores),
                "grade_errors":sum(s.get("grade_error") is not None for s in scores),
                "seconds":time.time()-started}
            (out/"progress.json").write_text(json.dumps(result,indent=2))
            print(json.dumps(result),flush=True)
    result["scope"]=f"fixed 128-problem MATH test subset; greedy; max {a.max_new_tokens} new tokens; boxed answer via math-verify 0.8.0"
    (out/"summary.json").write_text(json.dumps(result,indent=2))
if __name__=="__main__":main()
