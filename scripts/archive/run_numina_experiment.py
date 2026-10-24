import subprocess,json
from pathlib import Path
root=Path("/ssd/lzx/mopd");py=str(root/"mopd_train/bin/python")
with (root/"logs/numina-train.log").open("w") as f:
 r=subprocess.run([py,"-u","scripts/train_math_opd.py","--config","configs/numina_opd.json"],cwd=root,stdout=f,stderr=subprocess.STDOUT)
print("training exit",r.returncode,flush=True)
if r.returncode:raise SystemExit(r.returncode)
out=root/"runs/20261013-math-evaluation/numina_student_after"
with (root/"logs/numina-eval.log").open("w") as f:
 r=subprocess.run([py,"-u","scripts/eval_math_opd.py","--data",str(root/"data/math/eval.jsonl"),"--model",str(root/"models/Qwen3-4B-Base"),"--device","cuda:4","--adapter",str(root/"runs/20261013-numina-competition-opd/checkpoint-0064"),"--out",str(out)],cwd=root,stdout=f,stderr=subprocess.STDOUT)
print("evaluation exit",r.returncode,flush=True)
if r.returncode:raise SystemExit(r.returncode)
base=root/"runs/20261013-math-evaluation"
summary={k:json.loads((base/k/"summary.json").read_text()) for k in ["student_before","teacher","numina_student_after"]}
summary["accuracy_delta"]=summary["numina_student_after"]["accuracy"]-summary["student_before"]["accuracy"]
(base/"numina_comparison.json").write_text(json.dumps(summary,indent=2))
