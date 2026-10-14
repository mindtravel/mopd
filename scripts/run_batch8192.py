import subprocess,json
from pathlib import Path
r=Path("/ssd/lzx/mopd");py=str(r/"mopd_train/bin/python")
def launch(args,name):
 f=(r/"logs"/(name+".log")).open("w")
 p=subprocess.Popen([py,"-u",*args],cwd=r,stdout=f,stderr=subprocess.STDOUT)
 print(name,p.pid,flush=True);return p
ev=r/"runs/20261013-math-evaluation-8192";ev.mkdir(exist_ok=False)
common=["scripts/eval_math_opd.py","--data",str(r/"data/math/eval.jsonl"),"--max-new-tokens","8192"]
train=launch(["scripts/train_batched_opd.py","--config","configs/numina_8192_batch2.json"],"numina-8192-batch2-train")
before=launch(common+["--model",str(r/"models/Qwen3-4B-Base"),"--device","cuda:6","--out",str(ev/"student_before")],"math-8192-before")
teacher=launch(common+["--model",str(r/"models/Qwen3-4B"),"--device","cuda:7","--teacher","--out",str(ev/"teacher")],"math-8192-teacher")
rc=train.wait();print("training exit",rc,flush=True)
if rc:raise SystemExit(rc)
after=launch(common+["--model",str(r/"models/Qwen3-4B-Base"),"--device","cuda:2",
"--adapter",str(r/"runs/20261013-numina-8192-batch2-opd/checkpoint-0064"),"--out",str(ev/"numina_student_after")],"math-8192-after")
codes=[p.wait() for p in [before,teacher,after]]
(ev/"jobs.json").write_text(json.dumps(codes))
if any(codes):raise SystemExit(1)
result={k:json.loads((ev/k/"summary.json").read_text()) for k in ["student_before","teacher","numina_student_after"]}
result["accuracy_delta"]=result["numina_student_after"]["accuracy"]-result["student_before"]["accuracy"]
(ev/"comparison.json").write_text(json.dumps(result,indent=2))
