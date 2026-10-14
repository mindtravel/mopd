"""Single-domain OPD using a persisted, reproducible prompt pool plan."""
import argparse,json,time,random,traceback
from pathlib import Path
import torch
from transformers import AutoTokenizer
from mopd.backends.hf import load_pair
from mopd.training.dual_template import policy_loss
from mopd.data.domains import student_prompt,teacher_content
from mopd.data.templates import apply_chat_template_ids
from plan_domain_batches import make_plan
from mopd.training.batching import pack,logps,trim_generated,generation_eos_ids,is_eos_token
from train_math_opd import fingerprint,dump

def sync(c):
    for d in [c["student_device"],c["teacher_device"]]:torch.cuda.synchronize(d)

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);a=ap.parse_args()
 c=json.loads(Path(a.config).read_text());out=Path(c["output"]);out.mkdir(parents=True,exist_ok=False)
 dump(out/"config.json",c);dump(out/"status.json",{"state":"loading"})
 try:
  plan=make_plan(c)
  dump(out/"batch_plan.json",plan)
  torch.manual_seed(c["seed"]);tok,s,t=load_pair(c)
  stok=AutoTokenizer.from_pretrained(c["student"]);stok.padding_side="left"
  if stok.pad_token_id is None: stok.pad_token=stok.eos_token
  eos_ids=generation_eos_ids(s,stok)
  batch=c["micro_batch_size"];total=c["accumulation"];assert total%batch==0
  params=[p for p in s.parameters() if p.requires_grad]
  initial=[p.detach().cpu().clone() for p in params];th=fingerprint(t)
  opt=torch.optim.AdamW(params,lr=c["lr"])
  with (out/"metrics.jsonl").open("w",buffering=1) as mf,(out/"rollouts.jsonl").open("w",buffering=1) as rf:
   for step in range(c["steps"]):
    opt.zero_grad(set_to_none=True);sync(c);started=time.perf_counter()
    stats=[];times={"generation_seconds":0.,"teacher_seconds":0.,"student_seconds":0.}
    for offset in range(0,total,batch):
     items=plan[step][offset:offset+batch]
     prompt=stok([student_prompt(stok,r) for r in items],
       padding=True,return_tensors="pt").to(c["student_device"])
     dump(out/"status.json",{"state":"training","step":step,"micro":offset,"phase":"generating","batch_size":batch})
     s.eval();sync(c);mark=time.perf_counter()
     with torch.no_grad():
      ids=s.generate(**prompt,max_new_tokens=c["max_new_tokens"],do_sample=True,
       temperature=1.,top_k=0,top_p=1.,eos_token_id=list(eos_ids),
       pad_token_id=stok.pad_token_id,use_cache=True)
     sync(c);times["generation_seconds"]+=time.perf_counter()-mark
     answers=[trim_generated(ids[i,prompt.input_ids.shape[1]:],eos_ids) for i in range(batch)]
     seqs=[];starts=[];tseqs=[];tstarts=[]
     for i,(r,answer) in enumerate(zip(items,answers)):
      prefix=prompt.input_ids[i][prompt.attention_mask[i].bool()].tolist()
      seqs.append(prefix+answer);starts.append(len(prefix))
      prefix_t=apply_chat_template_ids(tok,[{"role":"user","content":teacher_content(r)}],add_generation_prompt=True,enable_thinking=False)
      ts=len(prefix_t);ti=torch.tensor([prefix_t+answer],device=c["teacher_device"])
      assert ti[0,ts:].tolist()==answer
      tseqs.append(ti[0].tolist());tstarts.append(ts)
     sid,sa,sm=pack(seqs,starts,stok.pad_token_id,c["student_device"])
     tid,ta,tm=pack(tseqs,tstarts,tok.pad_token_id,c["teacher_device"])
     dump(out/"status.json",{"state":"training","step":step,"micro":offset,"phase":"scoring"})
     sync(c);mark=time.perf_counter()
     with torch.no_grad():tp=[v.to(c["student_device"]) for v in logps(t,tid,ta,tm)]
     sync(c);times["teacher_seconds"]+=time.perf_counter()-mark
     s.train();mark=time.perf_counter()
     sp=logps(s,sid,sa,sm);losses=[]
     for i,(sv,tv) in enumerate(zip(sp,tp)):
      loss,info=policy_loss(sv,tv);assert torch.isfinite(loss)
      losses.append(loss)
      info.update(tokens=len(answers[i]),loss=loss.item(),truncated=not is_eos_token(answers[i][-1],eos_ids))
      stats.append(info)
      rf.write(json.dumps({"step":step+1,"micro":offset+i,"id":items[i]["id"],
       "domain":items[i]["domain"],"source":items[i].get("source"),"student_version":step,
       "response":stok.decode(answers[i]),"response_ids":answers[i],
       "student_prefix_tokens":starts[i],"teacher_prefix_tokens":tstarts[i],**info},ensure_ascii=False)+"\n")
     (torch.stack(losses).sum()/total).backward()
     sync(c);times["student_seconds"]+=time.perf_counter()-mark
     del sp,tp,losses,loss,sv,tv,sid,sa,sm,tid,ta,tm,ids,prompt
    grad=torch.nn.utils.clip_grad_norm_(params,1.)
    assert torch.isfinite(grad) and grad>0
    opt.step();sync(c)
    report={"step":step+1,"loss":sum(x["loss"] for x in stats)/total,
     "grad_norm":float(grad),"tokens":sum(x["tokens"] for x in stats),
     "truncated":sum(x["truncated"] for x in stats),"seconds":time.perf_counter()-started,**times}
    mf.write(json.dumps(report)+"\n");print(json.dumps(report),flush=True)
    dump(out/"status.json",{"state":"training",**report})
    if (step+1)%c["save_every"]==0 or step+1==c["steps"]:
     ckpt=out/f"checkpoint-{step+1:04d}";s.save_pretrained(ckpt,safe_serialization=True)
     torch.save({"step":step+1,"optimizer":opt.state_dict(),"rng":torch.get_rng_state(),
       "cuda_rng":torch.cuda.get_rng_state_all()},ckpt/"training_state.pt")
  delta=sum((p.detach().cpu()-v).float().square().sum().item() for p,v in zip(params,initial))**.5
  assert delta>0 and th==fingerprint(t) and all(p.grad is None for p in t.parameters())
  dump(out/"status.json",{"state":"complete","steps":c["steps"],"teacher_unchanged":True,
   "adapter_delta_l2":delta,"final_adapter":str(ckpt)})
 except BaseException:
  dump(out/"status.json",{"state":"failed","traceback":traceback.format_exc()});raise
if __name__=="__main__":main()
