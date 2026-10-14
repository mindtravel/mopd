"""Single frozen teacher, student LoRA with asymmetric generation/scoring templates."""
import argparse,json,time,hashlib
from pathlib import Path
import torch
from mopd.backends.hf import load_pair
from mopd.rollout.sampling import generate
from mopd.teacher.scoring import score_response
from mopd.training.losses import sampled_reverse_kl
from mopd.data.prompts import TRAIN,HELDOUT

def save(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def digest(m):
 h=hashlib.sha256()
 for n,p in m.named_parameters(): h.update(n.encode()); h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); a=ap.parse_args()
 c=json.loads(Path(a.config).read_text()); out=Path(c["output"]); out.mkdir(parents=True,exist_ok=False)
 save(out/"config.json",c); torch.manual_seed(c["seed"]); tok,student,teacher=load_pair(c)
 trainable={n:p for n,p in student.named_parameters() if p.requires_grad}
 assert trainable and all("lora_" in n for n in trainable)
 thash=digest(teacher); initial={n:p.detach().cpu().clone() for n,p in trainable.items()}
 opt=torch.optim.AdamW(trainable.values(),lr=c["lr"]); held=[]
 for i,prompt in enumerate(HELDOUT):
  ids,mask,text,stprompt=generate(student,tok,prompt,c["student_device"],c["max_new_tokens"],c["seed"]+100+i)
  tl,_,_,n=score_response(teacher,tok,prompt,text,ids[:,mask[0]],c["teacher_device"],c["student_device"])
  with torch.no_grad():
   sl=student(ids,use_cache=False).logits; sel=mask[:,1:]; sp=sl[:,:-1][sel].float().log_softmax(-1)
   targets=ids[:,1:][sel]; tp=tl[0,:n].float().log_softmax(-1); kl=(sp.exp()*(sp-tp)).sum(-1).mean().item()
  held.append((ids,mask,tl.cpu(),{"prompt":prompt,"response":text,"kl_before":kl})); del tl,sl
 with (out/"metrics.jsonl").open("w",buffering=1) as mf,(out/"rollouts.jsonl").open("w",buffering=1) as rf:
  for step in range(c["steps"]):
   start=time.time(); prompt=TRAIN[step%len(TRAIN)]
   ids,mask,text,stprompt=generate(student,tok,prompt,c["student_device"],c["max_new_tokens"],c["seed"]+step)
   tl,teacher_ids,teacher_start,n=score_response(teacher,tok,prompt,text,ids[:,mask[0]],c["teacher_device"],c["student_device"])
   student.train(); opt.zero_grad(set_to_none=True); logits=student(ids,use_cache=False).logits
   loss,stats=sampled_reverse_kl(logits,tl,ids,mask,n); assert torch.isfinite(loss); loss.backward()
   grad=torch.nn.utils.clip_grad_norm_(trainable.values(),1.0); assert torch.isfinite(grad); opt.step()
   row={"step":step+1,"student_version":step,"loss":loss.item(),"grad_norm":float(grad),"seconds":time.time()-start,**stats}
   mf.write(json.dumps(row)+"\n"); print(json.dumps(row),flush=True)
   rf.write(json.dumps({"step":step+1,"prompt":prompt,"student_prompt":stprompt,"response":text,"teacher_start":teacher_start,"teacher_input_ids":teacher_ids.tolist(),"input_ids":ids.cpu().tolist(),"response_mask":mask.cpu().tolist()},ensure_ascii=False)+"\n")
   del ids,mask,logits,tl,loss
 changed=sum((initial[n]-p.detach().cpu()).float().square().sum().item() for n,p in trainable.items())**0.5
 assert changed>0 and all(p.grad is None for p in teacher.parameters()) and thash==digest(teacher)
 student.save_pretrained(out/"adapter",safe_serialization=True); tok.save_pretrained(out/"tokenizer")
 save(out/"summary.json",{"status":"PASS","steps":c["steps"],"adapter_delta_l2":changed,"teacher_unchanged":True,"template_mode":"student-native generation; teacher-native response scoring","scope":"template-alignment OPD validation"})
if __name__=="__main__": main()
