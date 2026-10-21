"""Synchronous two-teacher OPD; shared on-policy student, fixed domain quotas."""
import argparse,hashlib,json,signal,sys,time,traceback
from pathlib import Path
import torch
from mopd.backends.multi_teacher import load_models
from mopd.backends.fairy2i_qat_adapter import save_fairy2i_qat_update
from mopd.data.mixture import plan_domains
from mopd.data.multidomain import domain_scale
from mopd.data.domains import teacher_content
from mopd.data.templates import apply_chat_template_ids
from mopd.rollout.hf_batch import generate
from mopd.rollout.replicas import RolloutReplicas
from mopd.rollout.processes import ProcessRollouts
from mopd.training.batching import pack,logps,generation_eos_ids,is_eos_token
from mopd.config import load
from mopd.training.dual_template import policy_loss
from mopd.evaluation.versioning import write_manifest
from mopd.training.schedule import build_scheduler
from train_math_opd import fingerprint,dump

def sync(c):
    for device in {c["student_device"],*(d["device"] for d in c["domains"].values())}:
        torch.cuda.synchronize(device)
def plan(c):
    return plan_domains(c)
def run(c):
    devices={c["student_device"],*c.get("rollout_devices",[]),*(d["device"] for d in c["domains"].values())}
    if "cuda:1" in devices:raise ValueError("GPU 1 is reserved")
    batches=plan(c)
    out=Path(c["output"]);out.mkdir(parents=True,exist_ok=False)
    dump(out/"config.json",c);dump(out/"batch_plan.json",batches)
    dump(out/"status.json",{"state":"loading"})
    write_manifest(c,out,{"state":"loading","step":0})
    try:
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
        with (out/"metrics.jsonl").open("w",buffering=1) as metrics,(out/"rollouts.jsonl").open("w",buffering=1) as rollouts:
            for step,rows in enumerate(batches,1):
                started=time.perf_counter();opt.zero_grad(set_to_none=True);lr=sched.get_last_lr()[0]
                dump(out/"status.json",{"state":"training","step":step,"phase":"generating"})
                if replicas:
                    trajectories,gen_seconds,rollout_stats=replicas.generate(student,rows,step-1)
                else:
                    trajectories,gen_seconds=generate(student,stok,rows,c["student_device"],c["max_new_tokens"],c["generation_batch_size"],prompt_format=c["student_prompt_format"])
                    rollout_stats={}
                stats={};teacher_seconds=0.;student_seconds=0.
                for domain,spec in c["domains"].items():
                    tt,teacher=teachers[domain]
                    selected=[r for r in trajectories if r["row"]["domain"]==domain]
                    entries=[];weighted_loss=0.
                    before=[p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
                    for offset in range(0,len(selected),c["micro_batch_size"]):
                        chunk=selected[offset:offset+c["micro_batch_size"]]
                        seqs=[r["prefix"]+r["answer"] for r in chunk]
                        starts=[len(r["prefix"]) for r in chunk]
                        tseqs=[];tstarts=[]
                        for r in chunk:
                            prefix=apply_chat_template_ids(tt,[{"role":"user","content":teacher_content(r["row"])}],add_generation_prompt=True,enable_thinking=False)
                            tseqs.append(prefix+r["answer"]);tstarts.append(len(prefix))
                            assert tseqs[-1][tstarts[-1]:]==r["answer"]
                        dump(out/"status.json",{"state":"training","step":step,"phase":"scoring","domain":domain})
                        tid,ta,tm=pack(tseqs,tstarts,tt.pad_token_id,spec["device"])
                        sync(c);mark=time.perf_counter()
                        with torch.no_grad():tp=[v.to(c["student_device"]) for v in logps(teacher,tid,ta,tm)]
                        sync(c);teacher_seconds+=time.perf_counter()-mark
                        del tid,ta,tm
                        sid,sa,sm=pack(seqs,starts,stok.pad_token_id,c["student_device"])
                        student.train();mark=time.perf_counter()
                        sp=logps(student,sid,sa,sm);losses=[]
                        for i,(sv,tv,r) in enumerate(zip(sp,tp,chunk)):
                            loss,info=policy_loss(sv,tv)
                            if not torch.isfinite(loss):raise ValueError("Nonfinite loss")
                            losses.append(loss)
                            info.update(loss=loss.item(),tokens=len(r["answer"]),truncated=not is_eos_token(r["answer"][-1],eos_ids))
                            entries.append(info)
                            rollouts.write(json.dumps({"step":step,"student_version":step-1,
                              "domain":domain,"teacher":spec["teacher"],"id":r["row"]["id"],
                              "prompt":r["row"]["prompt"],"response":stok.decode(r["answer"]),
                              "response_ids":r["answer"],**info},ensure_ascii=False)+"\n")
                        domain_loss=torch.stack(losses).sum()*scales[domain]
                        weighted_loss+=domain_loss.item();domain_loss.backward()
                        sync(c);student_seconds+=time.perf_counter()-mark
                        del sp,tp,losses,domain_loss,loss,sv,tv,sid,sa,sm
                    contribution=sum((p.grad-v).float().square().sum().item() for p,v in zip(params,before))**0.5
                    if not contribution>0:raise ValueError("Zero domain gradient: "+domain)
                    del before
                    stats[domain]={"grad_contribution_l2":contribution,"count":len(entries),"loss":sum(e["loss"] for e in entries)/len(entries),
                      "weighted_loss":weighted_loss,"sampled_log_ratio":sum(e["sampled_log_ratio"] for e in entries)/len(entries),
                      "clipped_fraction":sum(e["clipped_fraction"] for e in entries)/len(entries),
                      "tokens":sum(e["tokens"] for e in entries),"truncated":sum(e["truncated"] for e in entries)}
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
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);a=p.parse_args()
    signal.signal(signal.SIGTERM,lambda *_:sys.exit("SIGTERM"))
    run(load(a.config))
