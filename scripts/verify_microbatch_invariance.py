"""Check that the student gradient does not depend on the micro-batch size.

Both training entries reduce per sample (token mean) and weight samples independently, so
the full-step gradient is a sum over samples; the only path through which the micro-batch
size can matter is padding/packing inside mopd.training.batching.logps. This takes real
student rollouts (prompt + response token IDs) from a MOPD run, computes the gradient of
sum_i w_i * mean_t logp_i with micro-batch A, again with A (bf16 noise baseline) and with B,
and reports relative L2 error and cosine similarity. Lossless when |A-B| is within the
A-vs-A noise.

  CUDA_VISIBLE_DEVICES=5 python scripts/verify_microbatch_invariance.py \
      --run runs/<mopd run> --checkpoint checkpoint-0064 --step 1 --a 2 --b 4
"""
import argparse,json
from pathlib import Path
import torch
from mopd.students.registry import build_student
from mopd.training.batching import pack,logps
def grad(model,params,seqs,starts,weights,micro,pad,device):
    for p in params:p.grad=None
    for i in range(0,len(seqs),micro):
        ids,att,mask=pack(seqs[i:i+micro],starts[i:i+micro],pad,device)
        lp=logps(model,ids,att,mask)
        sum(w*(-v.mean()) for w,v in zip(weights[i:i+micro],lp)).backward()
    return torch.cat([p.grad.detach().float().flatten() for p in params])
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run",required=True);ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--step",type=int,default=1);ap.add_argument("--a",type=int,default=2);ap.add_argument("--b",type=int,default=4)
    ap.add_argument("--max-samples",type=int,default=16);a=ap.parse_args()
    run=Path(a.run);c=json.loads((run/"config.json").read_text())
    c={**c,"student_device":"cuda:0","student_adapter":str(run/a.checkpoint)}
    tok,model=build_student(c);model.train()
    from mopd.protocol import student_prompt
    rows=[json.loads(l) for l in open(run/"rollouts.jsonl")]
    rows=[r for r in rows if r["step"]==a.step][:a.max_samples]
    plan={x["id"]:x for x in json.loads((run/"batch_plan.json").read_text())[a.step-1]}
    seqs=[];starts=[];weights=[]
    for r in rows:
        prefix=tok(student_prompt(tok,plan[r["id"]],c["student_prompt_format"]),add_special_tokens=False).input_ids
        seqs.append(prefix+r["response_ids"]);starts.append(len(prefix));weights.append(1/len(rows))
    params=[p for p in model.parameters() if p.requires_grad];pad=tok.pad_token_id
    ga=grad(model,params,seqs,starts,weights,a.a,pad,"cuda:0")
    ga2=grad(model,params,seqs,starts,weights,a.a,pad,"cuda:0")
    gb=grad(model,params,seqs,starts,weights,a.b,pad,"cuda:0")
    rel=lambda x,y:float((x-y).norm()/y.norm());cos=lambda x,y:float(torch.nn.functional.cosine_similarity(x,y,dim=0))
    out={"samples":len(seqs),"tokens":sum(len(s)-st for s,st in zip(seqs,starts)),
         f"rel_l2 micro{a.a}_vs_micro{a.a} (noise)":rel(ga2,ga),f"rel_l2 micro{a.b}_vs_micro{a.a}":rel(gb,ga),
         f"cos micro{a.b}_vs_micro{a.a}":cos(gb,ga),"grad_norm":float(ga.norm())}
    out["lossless"]=out[f"rel_l2 micro{a.b}_vs_micro{a.a}"]<=max(3*out[f"rel_l2 micro{a.a}_vs_micro{a.a} (noise)"],1e-2)
    print(json.dumps(out,indent=2))
if __name__=="__main__":main()
