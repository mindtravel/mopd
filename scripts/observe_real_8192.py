"""Observe real rollout memory until a completed step contains an 8192-token answer."""
import json,os,subprocess,time
from pathlib import Path
root=Path("/ssd/lzx/mopd");run=root/"runs/20261013-dual-domain-batch32-mgpu-opd"
out=root/"logs/real-8192-memory.jsonl";deadline=time.time()+1200
with out.open("a",buffering=1) as f:
 while time.time()<deadline:
  supervisor=json.loads((root/"logs/dual-domain-supervisor-status.json").read_text())
  pid=supervisor.get("pid")
  if pid:os.kill(pid,0)
  status=json.loads((run/"status.json").read_text())
  raw=subprocess.check_output(["nvidia-smi","-i","6,7","--query-gpu=index,memory.used,memory.free,utilization.gpu","--format=csv,noheader,nounits"],text=True)
  devices=[dict(zip(["index","used_mib","free_mib","utilization"],map(int,line.split(",")))) for line in raw.strip().splitlines()]
  f.write(json.dumps({"time":time.time(),"status":status,"devices":devices})+"\n")
  rows=[json.loads(l) for l in (run/"rollouts.jsonl").read_text().splitlines()]
  long=[r for r in rows if r["tokens"]==8192]
  metrics=[json.loads(l) for l in (run/"metrics.jsonl").read_text().splitlines()]
  steps={r["step"] for r in metrics}
  found=[r for r in long if r["step"] in steps]
  if found:
   step=found[0]["step"];metric=next(m for m in metrics if m["step"]==step)
   result={"case_ids":[r["id"] for r in found if r["step"]==step],"completed_8192_step":metric}
   (root/"logs/real-8192-result.json").write_text(json.dumps(result,indent=2))
   print(json.dumps(result),flush=True);break
  if status["state"] in ("failed","complete"):
   print(json.dumps({"terminal":status,"observed_long_answers":len(long)}),flush=True);break
  time.sleep(10)
 else:print("Observation deadline reached; training was not restarted.",flush=True)
