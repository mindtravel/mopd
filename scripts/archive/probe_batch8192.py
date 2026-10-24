import json,time,torch
from pathlib import Path
from mopd.backends.hf import load_pair
from mopd.training.batching import pack,logps
from mopd.training.dual_template import policy_loss
c=json.loads(Path("configs/numina_8192_batch2.json").read_text())
tok,s,t=load_pair(c);s.train()
seq=(tok.encode("A mathematical argument follows. ")*2200)[:8704]
sid,sa,sm=pack([seq,seq],[512,512],tok.pad_token_id,c["student_device"])
tid,ta,tm=pack([seq,seq],[512,512],tok.pad_token_id,c["teacher_device"])
for d in [c["student_device"],c["teacher_device"]]:torch.cuda.reset_peak_memory_stats(d)
start=time.time()
with torch.no_grad():tp=[v.to(c["student_device"]) for v in logps(t,tid,ta,tm)]
sp=logps(s,sid,sa,sm)
loss=torch.stack([policy_loss(a,b)[0] for a,b in zip(sp,tp)]).mean();loss.backward()
params=[p for p in s.parameters() if p.requires_grad]
grad=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(grad) and grad>0
torch.optim.AdamW(params,lr=c["lr"]).step()
result={"status":"PASS","batch_size":2,"tokens_per_example":8704,"response_tokens":8192,"grad_norm":float(grad),
"seconds":time.time()-start,"peak_gib":{d:torch.cuda.max_memory_allocated(d)/2**30 for d in [c["student_device"],c["teacher_device"]]}}
Path("logs/probe-8192-batch2.json").write_text(json.dumps(result,indent=2));print(result,flush=True)
