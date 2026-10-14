"""Synchronous sampled-token OPD on disjoint MATH training prompts."""
import argparse,json,time,traceback,random,hashlib
from pathlib import Path
import torch
from transformers import AutoTokenizer
from mopd.backends.hf import load_pair
from mopd.training.dual_template import student_inputs,teacher_inputs,response_logps,policy_loss
from mopd.training.batching import generation_eos_ids,is_eos_token

def dump(path,value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2))
def fingerprint(model):
    h=hashlib.sha256()
    for p in model.parameters():h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);a=ap.parse_args()
    c=json.loads(Path(a.config).read_text());out=Path(c["output"]);out.mkdir(parents=True,exist_ok=False)
    dump(out/"config.json",c);dump(out/"status.json",{"state":"loading"})
    try:
        torch.manual_seed(c["seed"]);tok,student,teacher=load_pair(c)
        stok=AutoTokenizer.from_pretrained(c["student"])
        eos_ids=generation_eos_ids(student,stok)
        rows=[json.loads(s) for s in Path(c["train_file"]).read_text().splitlines()]
        random.Random(c["seed"]).shuffle(rows)
        assert c["steps"]*c["accumulation"]<=len(rows)
        params=[p for p in student.parameters() if p.requires_grad]
        opt=torch.optim.AdamW(params,lr=c["lr"])
        initial=[p.detach().cpu().clone() for p in params]
        before=fingerprint(teacher)
        with (out/"metrics.jsonl").open("w",buffering=1) as metrics,(out/"rollouts.jsonl").open("w",buffering=1) as rf:
            for step in range(c["steps"]):
                opt.zero_grad(set_to_none=True);started=time.time();stats=[]
                for micro in range(c["accumulation"]):
                    row=rows[step*c["accumulation"]+micro]
                    prompt=student_inputs(stok,row["problem"],c["student_device"])
                    student.eval()
                    with torch.no_grad():
                        ids=student.generate(input_ids=prompt,attention_mask=torch.ones_like(prompt),
                            do_sample=True,temperature=1.,top_k=0,top_p=1.,
                            max_new_tokens=c["max_new_tokens"],eos_token_id=list(eos_ids),
                            pad_token_id=stok.pad_token_id,use_cache=True)
                    response=ids[:,prompt.shape[1]:]
                    ti,ts=teacher_inputs(tok,row["problem"],response,c["teacher_device"])
                    assert torch.equal(ti[:,ts:].cpu(),response.cpu())
                    with torch.no_grad():
                        tl=teacher(ti,use_cache=False).logits
                        tp=response_logps(tl,ti,ts).to(c["student_device"])
                    del tl,ti
                    student.train()
                    sl=student(ids,use_cache=False).logits
                    sp=response_logps(sl,ids,prompt.shape[1])
                    loss,info=policy_loss(sp,tp)
                    assert torch.isfinite(loss)
                    (loss/c["accumulation"]).backward()
                    info.update(tokens=response.numel(),loss=loss.item(),
                        truncated=not is_eos_token(response[0,-1],eos_ids))
                    stats.append(info)
                    rf.write(json.dumps({"step":step+1,"micro":micro,"id":row["id"],
                        "response":stok.decode(response[0]),"response_ids":response[0].tolist(),
                        "student_prefix_tokens":prompt.shape[1],"teacher_prefix_tokens":ts,
                        **info},ensure_ascii=False)+"\n")
                    del sl,sp,tp,loss,ids,response
                grad=torch.nn.utils.clip_grad_norm_(params,1.)
                assert torch.isfinite(grad) and grad>0
                opt.step()
                report={"step":step+1,"grad_norm":float(grad),"seconds":time.time()-started,
                    "loss":sum(s["loss"] for s in stats)/len(stats),
                    "tokens":sum(s["tokens"] for s in stats),
                    "truncated":sum(s["truncated"] for s in stats)}
                metrics.write(json.dumps(report)+"\n");print(json.dumps(report),flush=True)
                dump(out/"status.json",{"state":"training",**report})
                if (step+1)%c["save_every"]==0 or step+1==c["steps"]:
                    ckpt=out/f"checkpoint-{step+1:04d}"
                    student.save_pretrained(ckpt,safe_serialization=True)
                    torch.save({"step":step+1,"optimizer":opt.state_dict(),"rng":torch.get_rng_state(),
                        "cuda_rng":torch.cuda.get_rng_state_all()},ckpt/"training_state.pt")
                    assert (ckpt/"adapter_config.json").is_file() and (ckpt/"adapter_model.safetensors").is_file()
        delta=sum((p.detach().cpu()-old).float().square().sum().item() for p,old in zip(params,initial))**.5
        assert delta>0 and all(p.grad is None for p in teacher.parameters())
        assert fingerprint(teacher)==before
        dump(out/"status.json",{"state":"complete","steps":c["steps"],"adapter_delta_l2":delta,
            "teacher_unchanged":True,"final_adapter":str(ckpt)})
    except BaseException:
        dump(out/"status.json",{"state":"failed","traceback":traceback.format_exc()});raise
if __name__=="__main__":main()
