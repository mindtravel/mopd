"""Temporarily borrow GPU3 between ability evaluations, then restore the watcher."""
import json,subprocess,time
from pathlib import Path
r=Path("/ssd/lzx/mopd")
out=r/"runs"/("vllm-adapter-smoke-"+str(int(time.time())))
status=r/"logs/vllm-smoke-status.json"
def save(**kw):status.write_text(json.dumps(dict(time=time.time(),output=str(out),**kw),indent=2))
try:
 raw=subprocess.check_output(["nvidia-smi","-i","3","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True)
 if int(raw.strip())<18000:raise RuntimeError("GPU3 no longer has enough free memory")
 with (r/"logs/vllm-adapter-smoke.log").open("w") as f:
  p=subprocess.Popen([str(r/"mopd_train/bin/python"),"-u","scripts/smoke_vllm_adapter.py","--device","cuda:3","--memory",".45","--out",str(out)],cwd=r,stdout=f,stderr=subprocess.STDOUT)
  save(state="running",pid=p.pid)
  rc=p.wait();save(state="complete" if rc==0 else "failed",returncode=rc)
finally:
 with (r/"logs/dual-ability-watcher.log").open("a") as f:
  p=subprocess.Popen([str(r/"mopd_train/bin/python"),"-u","scripts/watch_dual_ability.py"],cwd=r,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 (r/"logs/ability-watcher-restart.json").write_text(json.dumps({"pid":p.pid,"time":time.time(),"reason":"GPU3 vLLM smoke completed; resume checkpoint evaluations"}))
