"""One serial benchmark suite; no GPU 1; never interrupts existing GPU work."""
import json,subprocess,time,os,traceback
from pathlib import Path
ROOT=Path("/ssd/lzx/mopd")
STATUS=ROOT/"logs/vllm-probe-status.json"
OUT=ROOT/"runs"/("rollout-real-comparison-"+str(int(time.time())))
OUT.mkdir()
def status(**data):
 p=STATUS.with_suffix(".partial");p.write_text(json.dumps(dict(time=time.time(),suite=str(OUT),**data),indent=2));p.replace(STATUS)
def free_two():
 raw=subprocess.check_output(["nvidia-smi","-i","2","--query-gpu=memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True)
 mem,util=map(int,raw.strip().split(","))
 return mem<1000 and util<5
def wait_ready(vllm=False):
 deadline=time.time()+12*3600
 while time.time()<deadline:
  if vllm:
   check=subprocess.run([str(ROOT/"mopd_infer/bin/python"),"-c","import vllm; import transformers; assert transformers.__version__=='4.55.4'"],capture_output=True,text=True)
   if check.returncode:
    status(state="waiting_install",last_error=check.stderr[-1500:]);time.sleep(30);continue
  if free_two():return
  status(state="waiting_gpu",device=2);time.sleep(15)
 raise TimeoutError("GPU/environment not ready in 12 hours")
def run(name,python,script,args):
 wait_ready(vllm=name.startswith("vllm"))
 target=OUT/name
 with (OUT/(name+".log")).open("w") as log,(OUT/(name+"-memory.jsonl")).open("w",buffering=1) as memory:
  p=subprocess.Popen([str(ROOT/python),"-u",script,"--device","2","--out",str(target),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  status(state="benchmarking",name=name,pid=p.pid,device=2)
  peak=0
  while p.poll() is None:
   raw=subprocess.check_output(["nvidia-smi","-i","2","--query-gpu=memory.used,memory.total,utilization.gpu","--format=csv,noheader,nounits"],text=True)
   used,total,util=map(int,raw.strip().split(","))
   peak=max(peak,used);memory.write(json.dumps(dict(time=time.time(),used_mib=used,total_mib=total,utilization=util))+"\n")
   time.sleep(1)
 result=json.loads((target/"result.json").read_text()) if (target/"result.json").exists() else {"state":"failed"}
 result.update(returncode=p.returncode,whole_card_peak_mib=peak,name=name)
 return result
try:
 results=[]
 for name,python,script,args in [
  ("hf-b16","mopd_train/bin/python","scripts/benchmark_hf_real.py",["--batch","16"]),
  ("hf-b20","mopd_train/bin/python","scripts/benchmark_hf_real.py",["--batch","20"]),
  ("vllm-s16","mopd_infer/bin/python","scripts/benchmark_vllm_real.py",["--max-seqs","16"]),
  ("vllm-s32","mopd_infer/bin/python","scripts/benchmark_vllm_real.py",["--max-seqs","32"])]:
  results.append(run(name,python,script,args))
  (OUT/"comparison.json").write_text(json.dumps(results,indent=2))
 status(state="complete",results=results)
except BaseException:
 status(state="failed",error=traceback.format_exc());raise
