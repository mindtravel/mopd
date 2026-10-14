"""Wait for a genuinely free permitted GPU, then run the isolated real-prompt probe."""
import json,subprocess,time,os
from pathlib import Path
r=Path("/ssd/lzx/mopd")
status=r/"logs/vllm-probe-status.json"
deadline=time.time()+12*3600
def report(**kw):
 p=status.with_suffix(".partial");p.write_text(json.dumps(dict(time=time.time(),**kw),indent=2));p.replace(status)
while time.time()<deadline:
 check=subprocess.run([str(r/"mopd_infer/bin/python"),"-c","import vllm"],capture_output=True)
 if check.returncode:
  report(state="waiting_install");time.sleep(30);continue
 raw=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True)
 free=[int(v[0]) for line in raw.splitlines() if len(v:=line.split(","))==3 and int(v[0]) in (2,3) and int(v[1])<1000 and int(v[2])<5]
 if not free:
  report(state="waiting_gpu",allowed=[2,3]);time.sleep(30);continue
 gpu=free[0]
 out=r/"runs"/("vllm-real-probe-"+str(int(time.time())))
 report(state="benchmarking",device=gpu,output=str(out),scope="unadapted Base; greedy real batch; not a trained-adapter verification")
 with (r/"logs/vllm-real-probe.log").open("w") as log:
  p=subprocess.Popen([str(r/"mopd_infer/bin/python"),"-u","scripts/benchmark_vllm_real.py","--device",str(gpu),"--out",str(out)],cwd=r,stdout=log,stderr=subprocess.STDOUT)
  rc=p.wait()
 report(state="complete" if rc==0 else "failed",returncode=rc,device=gpu,output=str(out))
 break
else:report(state="timeout",reason="No available environment/GPU within 12 hours; no tasks stopped")
