"""Plain supervised fine-tuning on real (prompt, reference answer) pairs.

Same student construction (mopd.students.registry), same prompt protocol
(mopd.protocol.student_prompt) and same packed-logprob forward
(mopd.training.batching.pack/logps) as the MOPD OPD entries, so a checkpoint from
here is a drop-in comparison: same LoRA shape, same eval harness via
mopd.evaluation.versioning.write_manifest. The only difference from
scripts/train_domain_opd.py is that there is no rollout and no teacher: the
loss is plain token-level negative log-likelihood on a fixed reference answer.

Does not go through mopd.config.load: that schema requires a teacher and a
rollout backend per domain, which do not exist for SFT.
"""
import argparse,json,time,traceback
from pathlib import Path
import torch
from mopd.data.pool import build_plan,load_pool
from mopd.protocol import student_prompt
from mopd.students.registry import build_student,save_student
from mopd.training.batching import pack,logps
from mopd.evaluation.versioning import write_manifest
from mopd.runtime.artifacts import dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);a=ap.parse_args()
    c=json.loads(Path(a.config).read_text());out=Path(c["output"]);out.mkdir(parents=True,exist_ok=False)
    dump(out/"config.json",c);dump(out/"status.json",{"state":"loading"})
    try:
        torch.manual_seed(c["seed"])
        rows=load_pool(c["train_file"])
        plan=build_plan(rows,seed=c["seed"],steps=c["steps"],batch_size=c["accumulation"])
        dump(out/"batch_plan.json",plan)
        tok,model=build_student(c);model.train()  # gradient checkpointing only recomputes (instead of retaining) activations in train mode
        device=c["student_device"];pad=tok.pad_token_id;eos=tok.eos_token_id
        batch=c["micro_batch_size"];total=c["accumulation"];assert total%batch==0
        params=[p for p in model.parameters() if p.requires_grad]
        initial=[p.detach().cpu().clone() for p in params]
        opt=torch.optim.AdamW(params,lr=c["lr"])
        max_len=c.get("max_seq_len")
        with (out/"metrics.jsonl").open("w",buffering=1) as mf:
            for step in range(c["steps"]):
                opt.zero_grad(set_to_none=True);started=time.perf_counter()
                tokens=0;truncated=0;losses_report=[]
                for offset in range(0,total,batch):
                    items=plan[step][offset:offset+batch]
                    seqs=[];starts=[]
                    for row in items:
                        prefix=tok(student_prompt(tok,row,c["student_prompt_format"]),add_special_tokens=False).input_ids
                        response=tok(row["response"],add_special_tokens=False).input_ids+[eos]
                        seq=prefix+response
                        if max_len and len(seq)>max_len:
                            response=response[:max_len-len(prefix)]
                            if not response: continue
                            seq=prefix+response;truncated+=1
                        seqs.append(seq);starts.append(len(prefix))
                        tokens+=len(response)
                    if not seqs: continue
                    ids,attention,response_mask=pack(seqs,starts,pad,device)
                    lp=logps(model,ids,attention,response_mask)
                    micro_losses=[-v.mean() for v in lp]
                    for loss in micro_losses: losses_report.append(float(loss.detach()))
                    (torch.stack(micro_losses).sum()/total).backward()
                grad=torch.nn.utils.clip_grad_norm_(params,1.)
                assert torch.isfinite(grad)
                opt.step()
                report={"step":step+1,"loss":sum(losses_report)/len(losses_report) if losses_report else None,
                    "grad_norm":float(grad),"tokens":tokens,"truncated":truncated,
                    "seconds":time.perf_counter()-started}
                mf.write(json.dumps(report)+"\n");print(json.dumps(report),flush=True)
                dump(out/"status.json",{"state":"training",**report})
                if (step+1)%c["save_every"]==0 or step+1==c["steps"]:
                    ckpt=out/f"checkpoint-{step+1:04d}"
                    save_student(model,c,ckpt,step+1)
                    torch.save({"step":step+1,"optimizer":opt.state_dict(),"rng":torch.get_rng_state(),
                        "cuda_rng":torch.cuda.get_rng_state_all()},ckpt/"training_state.pt")
                    write_manifest(c,out,{"state":"training","step":step+1})
        delta=sum((p.detach().cpu()-v).float().square().sum().item() for p,v in zip(params,initial))**.5
        assert delta>0
        dump(out/"status.json",{"state":"complete","steps":c["steps"],
            "adapter_delta_l2":delta,"final_adapter":str(ckpt)})
        write_manifest(c,out,{"state":"complete","step":c["steps"]})
    except BaseException:
        dump(out/"status.json",{"state":"failed","traceback":traceback.format_exc()});raise

if __name__=="__main__":main()
