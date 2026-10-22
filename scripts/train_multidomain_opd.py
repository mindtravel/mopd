"""Synchronous two-teacher OPD; shared on-policy student, fixed domain quotas."""
import argparse,hashlib,json,signal,sys,time,traceback
from pathlib import Path
import torch
from mopd.backends.multi_teacher import load_models
from mopd.backends.fairy2i_qat_adapter import save_fairy2i_qat_update
from mopd.data.mixture import plan_domains
from mopd.data.multidomain import domain_scale
from mopd.rollout.hf_batch import generate
from mopd.rollout.replicas import RolloutReplicas
from mopd.rollout.processes import ProcessRollouts
from mopd.training.batching import generation_eos_ids
from mopd.config import load
from mopd.evaluation.versioning import write_manifest
from mopd.training.schedule import build_scheduler
from mopd.rollout.base import RolloutStream,open_stream
from mopd.training.streaming import DomainBatcher
from mopd.training.opd_step import OPDStepScorer
from mopd.runtime.provenance import record as record_provenance
from train_math_opd import fingerprint,dump

def dump_read(path):return Path(path).read_text()
def sha256_file(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def sync(c):
    for device in {c["student_device"],*(d["device"] for d in c["domains"].values())}:
        torch.cuda.synchronize(device)
def plan(c):
    return plan_domains(c)
# Runtime knobs a resume may change: they alter scheduling and batching only, not the objective,
# data, or optimizer (verified before use: greedy rollouts batch-invariant, gradients micro-batch-invariant).
RESUMABLE_OVERRIDES={"generation_batch_size":int,"micro_batch_size":int,"rollout_devices":lambda v:v.split(","),
                    "rollout_streaming":lambda v:{"true":True,"false":False}[v.lower()]}
def run(c,resume=None):
    """resume: {"checkpoint": Path, "overrides": {...}} continues an interrupted run in place."""
    devices={c["student_device"],*c.get("rollout_devices",[]),*(d["device"] for d in c["domains"].values())}
    if "cuda:1" in devices:raise ValueError("GPU 1 is reserved")
    batches=plan(c)
    out=Path(c["output"]);start=0
    if resume:
        ck=Path(resume["checkpoint"]);start=int(ck.name.rsplit("-",1)[1])
        if ck.parent.resolve()!=out.resolve():raise ValueError("checkpoint does not belong to this run")
        if json.loads(dump_read(out/"batch_plan.json"))!=json.loads(json.dumps(batches)):raise ValueError("re-planned batches differ from the run's batch_plan.json")
        # keep the interrupted run's full logs (never overwrite an earlier backup), then cut back to the checkpoint
        for name in ("metrics.jsonl","rollouts.jsonl"):
            f=out/name;raw=f.read_text();backup=out/f"{name}.before-resume-{start:04d}"
            if not backup.exists():backup.write_text(raw)
            f.write_text("".join(l+"\n" for l in raw.splitlines() if l.strip() and json.loads(l)["step"]<=start))
        # rollout adapters are written once per policy version (version = step - 1); move the interrupted
        # run's versions >= start aside so the resumed steps can write them again
        runtime=out/"vllm_runtime";stale=runtime/f"stale-before-resume-{start:04d}"
        for d in sorted(runtime.glob("adapter-*")) if runtime.exists() else []:
            if int(d.name.split("-")[1])>=start:
                stale.mkdir(exist_ok=True);d.rename(stale/d.name)
        c={**c,**resume["overrides"],"student_adapter":str(ck)}
        record_provenance(c,out,filename=f"provenance-resume-{start:04d}.json")
        dump(out/f"resume-{start:04d}.json",{"checkpoint":str(ck),"checkpoint_sha256":sha256_file(ck/"adapter_model.safetensors"),
            "overrides":resume["overrides"],"time":time.strftime("%Y-%m-%dT%H:%M:%S")})
    else:
        out.mkdir(parents=True,exist_ok=False)
        dump(out/"config.json",c);dump(out/"batch_plan.json",batches)
        dump(out/"status.json",{"state":"loading"})
        write_manifest(c,out,{"state":"loading","step":0})
    try:
        if not resume:record_provenance(c,out)
        torch.set_num_threads(2)
        torch.manual_seed(c["seed"])
        stok,student,teachers=load_models(c)
        eos_ids=generation_eos_ids(student,stok)
        backend=c.get("rollout_backend","replicas")
        factories={"processes":ProcessRollouts,"replicas":RolloutReplicas}
        if backend=="vllm":
            from mopd.rollout.vllm_processes import VllmRollouts
            factories["vllm"]=VllmRollouts
        if backend not in factories and backend != "hf":
            raise ValueError("Unknown rollout_backend: "+str(backend))
        replicas=factories[backend](c,student) if backend != "hf" and c.get("rollout_devices") else None
        params=[p for p in student.parameters() if p.requires_grad]
        initial=[p.detach().cpu().clone() for p in params]
        hashes={d:fingerprint(t) for d,(_,t) in teachers.items()}
        weights={d:s["weight"] for d,s in c["domains"].items()}
        counts={d:s["count"] for d,s in c["domains"].items()}
        scales={d:domain_scale(d,weights,counts) for d in counts}
        opt=torch.optim.AdamW(params,lr=c["lr"]);sched=build_scheduler(opt,c)
        if resume:
            state=torch.load(Path(resume["checkpoint"])/"training_state.pt",weights_only=False)
            if state["step"]!=start:raise ValueError("training_state step mismatch")
            opt.load_state_dict(state["optimizer"]);sched.load_state_dict(state["scheduler"])
            torch.set_rng_state(state["rng"]);torch.cuda.set_rng_state_all(state["cuda_rng"])
        mode="a" if resume else "w"
        with (out/"metrics.jsonl").open(mode,buffering=1) as metrics,(out/"rollouts.jsonl").open(mode,buffering=1) as rollouts:
            for step,rows in enumerate(batches,1):
                if step<=start:continue
                started=time.perf_counter();opt.zero_grad(set_to_none=True);lr=sched.get_last_lr()[0]
                dump(out/"status.json",{"state":"training","step":step,"phase":"generating"})
                # rollouts -> per-domain micro-batches (arrival order) -> teacher score + student backward
                if replicas:
                    stream=open_stream(replicas,student,rows,step-1,streaming=bool(c.get("rollout_streaming")))
                else:
                    trajectories,gen_seconds=generate(student,stok,rows,c["student_device"],c["max_new_tokens"],c["generation_batch_size"],prompt_format=c["student_prompt_format"])
                    stream=RolloutStream(trajectories);stream._started-=gen_seconds
                scorer=OPDStepScorer(config=c,student=student,student_tokenizer=stok,teachers=teachers,params=params,
                    scales=scales,eos_ids=eos_ids,sync=lambda:sync(c),rollout_log=rollouts,step=step)
                planned={d:sum(r["domain"]==d for r in rows) for d in c["domains"]}
                for domain,chunk in DomainBatcher(c["micro_batch_size"],planned).consume(stream):
                    scorer.process(domain,chunk)
                stats=scorer.domain_stats();teacher_seconds=scorer.teacher_seconds;student_seconds=scorer.student_seconds
                gen_seconds=stream.seconds;rollout_stats=stream.stats or {}
                grad=torch.nn.utils.clip_grad_norm_(params,1.)
                if not torch.isfinite(grad) or grad<=0:raise ValueError("Invalid student gradient")
                opt.step();sched.step();sync(c)
                report={"step":step,"lr":lr,"domains":stats,"loss":sum(s["weighted_loss"] for s in stats.values()),
                  "grad_norm":float(grad),"rollout":rollout_stats,"generation_seconds":gen_seconds,"teacher_seconds":teacher_seconds,
                  "student_seconds":student_seconds,"seconds":time.perf_counter()-started,
                  "generated_tokens_per_second":sum(s["tokens"] for s in stats.values())/gen_seconds,
                  "student_peak_gib":torch.cuda.max_memory_allocated(c["student_device"])/2**30}
                metrics.write(json.dumps(report)+"\n");print(json.dumps(report),flush=True)
                dump(out/"status.json",{"state":"training",**report})
                if step%c["save_every"]==0 or step==c["steps"]:
                    ckpt=out/f"checkpoint-{step:04d}"
                    if c.get("student_backend") == "fairy2i_qat":
                        save_fairy2i_qat_update(student,ckpt,base_model=c["student"],step=step)
                    else:
                        student.save_pretrained(ckpt,safe_serialization=True)
                    torch.save({"step":step,"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"rng":torch.get_rng_state(),
                      "cuda_rng":torch.cuda.get_rng_state_all()},ckpt/"training_state.pt")
                    write_manifest(c,out,{"state":"training","step":step})
        if hasattr(replicas,"close"):replicas.close()
        unchanged={d:hashes[d]==fingerprint(t) and all(p.grad is None for p in t.parameters()) for d,(_,t) in teachers.items()}
        delta=sum((p.detach().cpu()-v).float().square().sum().item() for p,v in zip(params,initial))**.5
        assert all(unchanged.values()) and delta>0
        dump(out/"status.json",{"state":"complete","steps":c["steps"],"teachers_unchanged":unchanged,
          "adapter_delta_l2":delta,"final_adapter":str(ckpt)})
        write_manifest(c,out,{"state":"complete","steps":c["steps"]})
    except BaseException as error:
        if "replicas" in locals() and hasattr(replicas,"close"):replicas.close()
        state="stopped" if isinstance(error,(SystemExit,KeyboardInterrupt)) else "failed"
        dump(out/"status.json",{"state":state,"traceback":traceback.format_exc()})
        write_manifest(c,out,{"state":state})
        raise
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True)
    p.add_argument("--resume-from",help="checkpoint directory of this run to continue from")
    p.add_argument("--override",action="append",default=[],help=f"KEY=VALUE, KEY in {sorted(RESUMABLE_OVERRIDES)} (resume only)")
    a=p.parse_args()
    if a.override and not a.resume_from:p.error("--override is only allowed with --resume-from")
    overrides={}
    for item in a.override:
        k,_,v=item.partition("=")
        if k not in RESUMABLE_OVERRIDES:p.error(f"{k} is not a resumable runtime knob")
        overrides[k]=RESUMABLE_OVERRIDES[k](v)
    signal.signal(signal.SIGTERM,lambda *_:sys.exit("SIGTERM"))
    if a.resume_from:
        c=load(Path(a.resume_from).parent/"config.json")  # the run's own recorded config, not the spec copy
        run(c,{"checkpoint":a.resume_from,"overrides":overrides})
    else:
        run(load(a.config))
