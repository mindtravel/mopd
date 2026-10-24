import json,time,torch
from pathlib import Path
from mopd.backends.hf import load_pair
from mopd.training.long_context import selected_logps
from mopd.training.dual_template import policy_loss
c=json.loads(Path("configs/numina_8192.json").read_text())
tok,s,t=load_pair(c);s.train()
# Deliberately fill the response budget, even if a natural answer ends early.
ids=torch.tensor([tok.encode("A mathematical argument follows. ")*2200],device=c["student_device"])[:,:8704]
assert ids.shape[1]==8704
for d in [c["student_device"],c["teacher_device"]]:torch.cuda.reset_peak_memory_stats(d)
now=time.time()
with torch.no_grad():tp=selected_logps(t,ids.to(c["teacher_device"]),512).to(c["student_device"])
sp=selected_logps(s,ids,512);loss,_=policy_loss(sp,tp);loss.backward()
params=[p for p in s.parameters() if p.requires_grad]
grad=torch.nn.utils.clip_grad_norm_(params,1.)
assert torch.isfinite(grad) and grad>0
torch.optim.AdamW(params,lr=c["lr"]).step()
r={"status":"PASS","response_tokens":8192,"total_tokens":8704,"grad_norm":float(grad),"seconds":time.time()-now,
"peak_gib":{d:torch.cuda.max_memory_allocated(d)/2**30 for d in [c["student_device"],c["teacher_device"]]}}
Path("logs/probe-8192.json").write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
