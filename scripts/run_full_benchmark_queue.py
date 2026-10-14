"""Serial GPU3 evaluation queue. Wait for existing baseline, preserve training GPUs."""
import json,time,subprocess,os,fcntl
from pathlib import Path
R=Path("/ssd/lzx/mopd");status=R/"logs/full-benchmark-queue.json"
def report(**kw):status.write_text(json.dumps(dict(time=time.time(),**kw),indent=2))
def job(name,model,data,teacher=False,adapter=None):
 out=R/"runs/full-eval"/name
 if (out/"summary.json").exists():return
 args=[str(R/"mopd_infer/bin/python"),"-u","scripts/generate_benchmark_vllm.py","--model",str(R/"models"/model),"--data",str(R/data),"--device","3","--max-seqs","32","--out",str(out)]
 if teacher:args.append("--teacher")
 if adapter:args+=["--adapter",str(adapter)]
 with (R/"logs"/(name+".log")).open("a") as f:
  p=subprocess.Popen(args,cwd=R,stdout=f,stderr=subprocess.STDOUT)
  report(state="running",name=name,pid=p.pid)
  if p.wait():raise RuntimeError("Generation failed: "+name)
  grading=subprocess.Popen([str(R/"mopd_train/bin/python"),"-u","scripts/grade_benchmark_outputs.py","--data",str(R/data),"--out",str(out)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
  report(state="grading",name=name,pid=grading.pid)
  if grading.wait():raise RuntimeError("Grading failed: "+name)
try:
 queue_lock=(R/"logs/full-benchmark-queue.lock").open("a")
 fcntl.flock(queue_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 existing=Path("/proc/625803")
 while existing.exists():
  cmd=(existing/"cmdline").read_bytes()
  if b"generate_benchmark_vllm.py" not in cmd:break
  report(state="waiting_existing_generation",pid=625803)
  time.sleep(30)
 job("student-base-ifeval541-vllm","Qwen3-4B-Base","data/instruction/ifeval.jsonl")
 job("student-base-math5000","Qwen3-4B-Base","data/full_eval/math-5000.jsonl")
 job("math-teacher-math5000","Qwen3-4B","data/full_eval/math-5000.jsonl",True)
 job("instruction-teacher-ifeval541","Qwen3-4B-Instruct-2507","data/instruction/ifeval.jsonl",True)
 run=R/"runs/20261014-dual-domain-batch32-mgpu-opd-b8"
 ck=run/"checkpoint-0064"
 while not (ck/"training_state.pt").exists():
  state=json.loads((run/"status.json").read_text())
  if state.get("state")=="failed":raise RuntimeError("Training failed; final model unavailable")
  report(state="waiting_final_checkpoint");time.sleep(30)
 job("student-final-math5000","Qwen3-4B-Base","data/full_eval/math-5000.jsonl",adapter=ck)
 job("student-final-ifeval541","Qwen3-4B-Base","data/instruction/ifeval.jsonl",adapter=ck)
 report(state="complete")
except BaseException as e:report(state="failed",error=repr(e));raise
