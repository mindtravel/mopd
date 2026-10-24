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

Multi-GPU: launch with `torchrun --standalone --nproc_per_node=N scripts/train_sft.py
--config ...` to data-parallelize one run across N GPUs; a plain `python scripts/train_sft.py
--config ...` (no torchrun) is unchanged single-GPU behavior, reading config["student_device"]
exactly as before -- every distributed code path below is gated on world_size > 1 and is a no-op
otherwise, so existing single-GPU callers (scripts/run_experiment.py) need no changes.

Design: NOT torch.nn.parallel.DistributedDataParallel. logps() (mopd/training/batching.py)
deliberately bypasses the student's top-level forward to chunk the LM head manually, so nothing
ever calls a module whose forward DDP could hook; wrapping in DDP would silently not synchronize
gradients. Instead each rank runs the unmodified single-GPU forward/backward on its own shard of
the step's batch, then every trainable parameter's .grad is summed (not averaged) across ranks:
every rank already scales its per-sample loss by the *global* per-step domain weight (computed
from the full, unsharded batch), so summing the disjoint per-rank shards reproduces exactly the
gradient a single GPU would compute over the whole step -- multi-GPU training is numerically the
same experiment, just sharded, not a different setting. Reported per-step metrics are likewise
summed across ranks so they describe the whole step. Only rank 0 writes files (metrics, status,
checkpoints, manifest); every rank still runs the full compute loop to stay in lockstep, since
each holds its own replica of the model and optimizer and needs to apply the same update.
"""
import argparse,json,os,time,traceback
from pathlib import Path
import torch
import torch.distributed as dist
from mopd.data.mixture import resolve_domain,plan_domains
from mopd.data.token_budget import plan_by_tokens
from mopd.protocol import student_prompt
from mopd.students.registry import build_student,save_student,is_sharded
from mopd.training import fsdp
from mopd.training.distributed import dist_info
from mopd.training.batching import pack,logps
from mopd.training.schedule import build_scheduler,describe
from mopd.evaluation.versioning import write_manifest
from mopd.runtime.artifacts import dump
from mopd.runtime.provenance import record as record_provenance
from mopd.runtime.profile import ModelShape,StepProfile

def encode(tok,row,prompt_format,eos,max_len):
    prefix=tok(student_prompt(tok,row,prompt_format),add_special_tokens=False).input_ids
    response=tok(row["response"],add_special_tokens=False).input_ids+[eos]
    if max_len and len(prefix)+len(response)>max_len:
        return prefix,response[:max_len-len(prefix)],True
    return prefix,response,False

def sync_stats(stats,device):
    """Sum each domain's local sample/loss/token/truncated counters across ranks: every rank only
    processed a shard of the step, so the reported metrics must cover the whole step, not one shard."""
    domains=sorted(stats)
    packed=torch.tensor([[stats[d]["samples"],stats[d]["loss_sum"],stats[d]["tokens"],stats[d]["truncated"]]
                         for d in domains],device=device,dtype=torch.float64)
    dist.all_reduce(packed,op=dist.ReduceOp.SUM)
    return {d:{"samples":int(packed[i,0]),"loss_sum":float(packed[i,1]),"tokens":int(packed[i,2]),
               "truncated":int(packed[i,3])} for i,d in enumerate(domains)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);a=ap.parse_args()
    c=json.loads(Path(a.config).read_text());out=Path(c["output"])
    rank,world_size,dist_device=dist_info()
    if dist_device:c["student_device"]=dist_device
    main_rank=rank==0
    for name,spec in c["domains"].items():resolve_domain(name,spec)
    describe(c)  # pure validation, no side effect; run on every rank
    if main_rank:
        out.mkdir(parents=True,exist_ok=False)
        dump(out/"config.json",c);dump(out/"status.json",{"state":"loading"})
    if world_size>1:
        dist.barrier()  # other ranks wait for `out` to exist before training, though none read it
    try:
        if main_rank:record_provenance(c,out)
        if world_size>1:dist.barrier()
        torch.manual_seed(c["seed"])  # identical on every rank: same LoRA init before training starts
        sharded=is_sharded(c)
        if sharded and world_size<4:raise ValueError("full fine-tuning (fp32 weights + AdamW) needs >= 4 GPUs to fit")
        tok,model=build_student(c);model.train()  # gradient checkpointing only recomputes (instead of retaining) activations in train mode
        scorer=fsdp.shard(model,**c.get("fsdp",{})) if sharded else model  # what logps() runs and the optimizer updates
        device=c["student_device"];pad=tok.pad_token_id;eos=tok.eos_token_id
        max_len=c.get("max_seq_len")
        if c.get("token_budget"):
            plan=plan_by_tokens(c,lambda row:len(encode(tok,row,c["student_prompt_format"],eos,max_len)[1]))
        else:
            plan=plan_domains(c)  # identical on every rank: deterministic from config["seed"] + data files
        if world_size>1 and any(len(rows)<world_size for rows in plan):
            raise ValueError(f"samples_per_step is smaller than world_size={world_size}: some ranks would get no rows")
        if main_rank:dump(out/"batch_plan.json",plan)
        weights={d:s["weight"] for d,s in c["domains"].items()}
        micro=c["micro_batch_size"]
        params=[p for p in scorer.parameters() if p.requires_grad]
        # sharded: each rank keeps its own shard and the norms are summed over ranks at the end
        initial=[fsdp.local(p).detach().cpu().clone() for p in params] if main_rank or sharded else None
        opt=torch.optim.AdamW(params,lr=c["lr"]);sched=build_scheduler(opt,c)
        mf=(out/"metrics.jsonl").open("w",buffering=1) if main_rank else None
        ckpt=None
        shape=ModelShape.of(scorer);recompute=c.get("gradient_checkpointing",True)
        try:
            for step,rows in enumerate(plan,1):
                prof=StepProfile(shape,world_size,recompute=recompute,device=device)
                opt.zero_grad(set_to_none=True);started=time.perf_counter();lr=sched.get_last_lr()[0]
                stats={d:{"samples":0,"loss_sum":0.,"tokens":0,"truncated":0} for d in weights}
                # Per-sample weight = domain weight / that domain's samples this step: per-sample token
                # mean, then domain mean, then domain weights (MOPD's domain_scale when counts are fixed).
                # Computed from the full step (all ranks' rows), not the local shard below, so summed
                # per-rank gradients reproduce the single-GPU gradient exactly (see module docstring).
                counts={d:sum(r["domain"]==d for r in rows) for d in weights}
                scales={d:weights[d]/counts[d] for d in weights}
                local_rows=rows[rank::world_size]  # disjoint, deterministic shard of this step's batch
                # Encode the shard once and micro-batch it by length: each micro batch pads to its longest
                # sequence, and in arrival order half the step's tokens were padding (profile: 0.50). The
                # gradient does not depend on the grouping (fixed per-sample scales, summed), only the work.
                encoded=sorted((encode(tok,row,c["student_prompt_format"],eos,max_len)+(row,) for row in local_rows),
                               key=lambda e:len(e[0])+len(e[1]),reverse=True)
                for offset in range(0,len(encoded),micro):
                    chunk=encoded[offset:offset+micro];items=[e[3] for e in chunk];seqs=[];starts=[];truncated=[]
                    for prefix,response,cut,row in chunk:
                        if not response:raise ValueError(f"prompt alone exceeds max_seq_len: {row['id']}")
                        seqs.append(prefix+response);starts.append(len(prefix));truncated.append(cut)
                    ids,attention,response_mask=pack(seqs,starts,pad,device)
                    prof.add_batch([len(q) for q in seqs],ids.shape[1])
                    with prof.phase("forward_backward"):
                        per_sample=[-v.mean() for v in logps(scorer,ids,attention,response_mask)]
                        loss=sum(scales[r["domain"]]*l for r,l in zip(items,per_sample))
                        loss.backward()
                    for r,l,s,start,cut in zip(items,per_sample,seqs,starts,truncated):
                        d=stats[r["domain"]];d["samples"]+=1;d["loss_sum"]+=float(l.detach())
                        d["tokens"]+=len(s)-start;d["truncated"]+=int(cut)
                peak=torch.tensor(torch.cuda.max_memory_allocated(device)/2**30,device=device)
                # whole-step token accounting for MFU: every rank's real tokens, padded widths and squared lengths
                lengths=torch.tensor([sum(prof.lengths),prof.padded,sum(q*q for q in prof.lengths)],device=device,dtype=torch.float64)
                with prof.phase("gradient_sync"):
                    if world_size>1:
                        dist.all_reduce(peak,op=dist.ReduceOp.MAX)  # the rank closest to OOM, not rank 0
                        dist.all_reduce(lengths)
                        if not sharded:  # FSDP already reduce-scattered sharded gradients (as a SUM, see fsdp.py)
                            for p in params:dist.all_reduce(p.grad,op=dist.ReduceOp.SUM)
                        stats=sync_stats(stats,device)
                with prof.phase("optimizer"):
                    grad=torch.nn.utils.clip_grad_norm_(params,1.)
                    grad=grad.full_tensor() if hasattr(grad,"full_tensor") else grad
                    if not torch.isfinite(grad) or grad<=0:raise ValueError("Invalid student gradient")
                    opt.step();sched.step()
                if main_rank:
                    domains={d:{"loss":s["loss_sum"]/s["samples"],"samples":s["samples"],"tokens":s["tokens"],
                                "truncated":s["truncated"]} for d,s in stats.items()}
                    report={"step":step,"lr":lr,"loss":sum(weights[d]*v["loss"] for d,v in domains.items()),
                        "grad_norm":float(grad),"domains":domains,
                        "tokens":sum(v["tokens"] for v in domains.values()),
                        "truncated":sum(v["truncated"] for v in domains.values()),
                        "seconds":time.perf_counter()-started,"world_size":world_size,
                        "max_memory_gib":round(float(peak),2),
                        "profile":prof.finish(total_tokens=int(lengths[0]),padded=int(lengths[1]),sum_sq=float(lengths[2]))}
                    mf.write(json.dumps(report)+"\n");print(json.dumps(report),flush=True)
                    dump(out/"status.json",{"state":"training",**report})
                if step%c["save_every"]==0 or step==c["steps"]:
                    ckpt=out/f"checkpoint-{step:04d}"
                    if sharded or main_rank:save_student(scorer if sharded else model,c,ckpt,step,main_rank)
                    if main_rank:
                        if not sharded:  # a sharded optimizer's rank-0 state is a fragment; runs never resume anyway
                            torch.save({"step":step,"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),
                                "rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all()},ckpt/"training_state.pt")
                        write_manifest(c,out,{"state":"training","step":step})
                if world_size>1:dist.barrier()
        finally:
            if mf:mf.close()
        if main_rank or sharded:
            delta2=sum((fsdp.local(p).detach().cpu()-v).float().square().sum().item() for p,v in zip(params,initial))
            if sharded:
                total=torch.tensor(delta2,device=device,dtype=torch.float64);dist.all_reduce(total);delta2=float(total)
        if main_rank:
            delta=delta2**.5
            assert delta>0
            dump(out/"status.json",{"state":"complete","steps":c["steps"],
                "adapter_delta_l2":delta,"final_adapter":str(ckpt)})
            write_manifest(c,out,{"state":"complete","step":c["steps"]})
    except BaseException:
        if main_rank:dump(out/"status.json",{"state":"failed","traceback":traceback.format_exc()})
        raise
    finally:
        if world_size>1:dist.destroy_process_group()

if __name__=="__main__":main()
