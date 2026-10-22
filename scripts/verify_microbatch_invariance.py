"""Check that the student gradient does not depend on the micro-batch size.

Both training entries reduce per sample (token mean) and weight samples independently, so
the full-step gradient is a sum over samples; the only path through which the micro-batch
size can matter is padding/packing inside mopd.training.batching.logps. This takes real
student rollouts (prompt + response token IDs) from a MOPD run and computes the gradient of
sum_i w_i * mean_t logp_i with micro-batch 1, A (the current setting) and B.

bf16 kernels round differently for different batch shapes, so even unpadded samples change at
the 1e-1 per-token-logprob level when the micro-batch changes; repeating A only measures
run-to-run noise at a fixed shape and is too strict a baseline. Lossless means B is no further
from micro-batch 1 than A already is: rel(B,1) <= 1.5 * rel(A,1).

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
    g1=grad(model,params,seqs,starts,weights,1,pad,"cuda:0")
    ga=grad(model,params,seqs,starts,weights,a.a,pad,"cuda:0")
    ga2=grad(model,params,seqs,starts,weights,a.a,pad,"cuda:0")
    gb=grad(model,params,seqs,starts,weights,a.b,pad,"cuda:0")
    rel=lambda x,y:float((x-y).norm()/y.norm());cos=lambda x,y:float(torch.nn.functional.cosine_similarity(x,y,dim=0))
    out={"samples":len(seqs),"tokens":sum(len(s)-st for s,st in zip(seqs,starts)),
         f"rel_l2 micro{a.a}_vs_micro{a.a} (same-shape noise)":rel(ga2,ga),
         f"rel_l2 micro{a.a}_vs_micro1 (current setting)":rel(ga,g1),f"rel_l2 micro{a.b}_vs_micro1":rel(gb,g1),
         f"rel_l2 micro{a.b}_vs_micro{a.a}":rel(gb,ga),f"cos micro{a.b}_vs_micro1":cos(gb,g1),f"cos micro{a.a}_vs_micro1":cos(ga,g1),
         "grad_norm":float(g1.norm())}
    out["lossless"]=rel(gb,g1)<=1.5*rel(ga,g1)
    print(json.dumps(out,indent=2))
if __name__=="__main__":main()
