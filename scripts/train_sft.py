"""Plain supervised fine-tuning on real (prompt, reference answer) pairs.

Everything except the training signal is shared with the MOPD entry
(scripts/train_multidomain_opd.py), so a comparison isolates the method:
  - batches: mopd.data.mixture.resolve_domain + plan_domains (fixed per-step domain
    and source quotas, no replacement, holdout overlap rejected before training);
  - loss weighting: per-sample token mean, then per-domain mean, then domain weights,
    the reduction MOPD applies via mopd.data.multidomain.domain_scale;
  - optional "token_budget": {"per_step": N} sizes each step by trained tokens instead of
    sample count (mopd.data.token_budget), same pools and shuffle order;
  - optimizer and schedule: AdamW + mopd.training.schedule, grad clip 1.0;
  - student, prompt protocol, packed forward, checkpoint and manifest.
The only difference is that there is no rollout and no teacher: each sample's loss
is token-level negative log-likelihood of its fixed reference answer.

Does not go through mopd.config.load: that schema requires a teacher and a rollout
backend per domain, which do not exist for SFT.
"""
import argparse,json,time,traceback
from pathlib import Path
import torch
from mopd.data.mixture import resolve_domain,plan_domains
from mopd.data.token_budget import plan_by_tokens
from mopd.protocol import student_prompt
from mopd.students.registry import build_student,save_student
from mopd.training.batching import pack,logps
from mopd.training.schedule import build_scheduler,describe
from mopd.evaluation.versioning import write_manifest
from mopd.runtime.artifacts import dump
from mopd.runtime.provenance import record as record_provenance

def encode(tok,row,prompt_format,eos,max_len):
    prefix=tok(student_prompt(tok,row,prompt_format),add_special_tokens=False).input_ids
    response=tok(row["response"],add_special_tokens=False).input_ids+[eos]
    if max_len and len(prefix)+len(response)>max_len:
        return prefix,response[:max_len-len(prefix)],True
    return prefix,response,False

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);a=ap.parse_args()
    c=json.loads(Path(a.config).read_text());out=Path(c["output"])
    for name,spec in c["domains"].items():resolve_domain(name,spec)
    describe(c)
    out.mkdir(parents=True,exist_ok=False)
    dump(out/"config.json",c);dump(out/"status.json",{"state":"loading"})
    try:
        record_provenance(c,out)
        torch.manual_seed(c["seed"])
        tok,model=build_student(c);model.train()  # gradient checkpointing only recomputes (instead of retaining) activations in train mode
        device=c["student_device"];pad=tok.pad_token_id;eos=tok.eos_token_id
        max_len=c.get("max_seq_len")
        if c.get("token_budget"):
            plan=plan_by_tokens(c,lambda row:len(encode(tok,row,c["student_prompt_format"],eos,max_len)[1]))
        else:
            plan=plan_domains(c)
        dump(out/"batch_plan.json",plan)
        weights={d:s["weight"] for d,s in c["domains"].items()}
        micro=c["micro_batch_size"]
        params=[p for p in model.parameters() if p.requires_grad]
        initial=[p.detach().cpu().clone() for p in params]
        opt=torch.optim.AdamW(params,lr=c["lr"]);sched=build_scheduler(opt,c)
        with (out/"metrics.jsonl").open("w",buffering=1) as mf:
            for step,rows in enumerate(plan,1):
                opt.zero_grad(set_to_none=True);started=time.perf_counter();lr=sched.get_last_lr()[0]
                stats={d:{"samples":0,"loss_sum":0.,"tokens":0,"truncated":0} for d in weights}
                # Per-sample weight = domain weight / that domain's samples this step: per-sample token
                # mean, then domain mean, then domain weights (MOPD's domain_scale when counts are fixed).
                counts={d:sum(r["domain"]==d for r in rows) for d in weights}
                scales={d:weights[d]/counts[d] for d in weights}
                for offset in range(0,len(rows),micro):
                    items=rows[offset:offset+micro];seqs=[];starts=[];truncated=[]
                    for row in items:
                        prefix,response,cut=encode(tok,row,c["student_prompt_format"],eos,max_len)
                        if not response:raise ValueError(f"prompt alone exceeds max_seq_len: {row['id']}")
                        seqs.append(prefix+response);starts.append(len(prefix));truncated.append(cut)
                    ids,attention,response_mask=pack(seqs,starts,pad,device)
                    per_sample=[-v.mean() for v in logps(model,ids,attention,response_mask)]
                    loss=sum(scales[r["domain"]]*l for r,l in zip(items,per_sample))
                    loss.backward()
                    for r,l,s,start,cut in zip(items,per_sample,seqs,starts,truncated):
                        d=stats[r["domain"]];d["samples"]+=1;d["loss_sum"]+=float(l.detach())
                        d["tokens"]+=len(s)-start;d["truncated"]+=int(cut)
                grad=torch.nn.utils.clip_grad_norm_(params,1.)
                if not torch.isfinite(grad) or grad<=0:raise ValueError("Invalid student gradient")
                opt.step();sched.step()
                domains={d:{"loss":s["loss_sum"]/s["samples"],"samples":s["samples"],"tokens":s["tokens"],
                            "truncated":s["truncated"]} for d,s in stats.items()}
                report={"step":step,"lr":lr,"loss":sum(weights[d]*v["loss"] for d,v in domains.items()),
                    "grad_norm":float(grad),"domains":domains,
                    "tokens":sum(v["tokens"] for v in domains.values()),
                    "truncated":sum(v["truncated"] for v in domains.values()),
                    "seconds":time.perf_counter()-started}
                mf.write(json.dumps(report)+"\n");print(json.dumps(report),flush=True)
                dump(out/"status.json",{"state":"training",**report})
                if step%c["save_every"]==0 or step==c["steps"]:
                    ckpt=out/f"checkpoint-{step:04d}"
                    save_student(model,c,ckpt,step)
                    torch.save({"step":step,"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),
                        "rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all()},ckpt/"training_state.pt")
                    write_manifest(c,out,{"state":"training","step":step})
        delta=sum((p.detach().cpu()-v).float().square().sum().item() for p,v in zip(params,initial))**.5
        assert delta>0
        dump(out/"status.json",{"state":"complete","steps":c["steps"],
            "adapter_delta_l2":delta,"final_adapter":str(ckpt)})
        write_manifest(c,out,{"state":"complete","step":c["steps"]})
    except BaseException:
        dump(out/"status.json",{"state":"failed","traceback":traceback.format_exc()});raise

if __name__=="__main__":main()
