"""Launch isolated baseline evaluations and training; evaluate final saved adapter."""
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path("/ssd/lzx/mopd")
RUN=ROOT/"runs/20261013-math-level345-opd"
EVAL=ROOT/"runs/20261013-math-evaluation"
PY=str(ROOT/"mopd_train/bin/python")
def launch(args,name):
    log=open(ROOT/"logs"/(name+".log"),"w")
    p=subprocess.Popen([PY,"-u",*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    print(json.dumps({"job":name,"pid":p.pid}),flush=True)
    return p
def main():
    EVAL.mkdir(exist_ok=False)
    common=["scripts/eval_math_opd.py","--data",str(ROOT/"data/math/eval.jsonl")]
    jobs=[]
    jobs.append(launch(common+["--model",str(ROOT/"models/Qwen3-4B-Base"),"--device","cuda:6",
        "--out",str(EVAL/"student_before")],"math-student-before"))
    jobs.append(launch(common+["--model",str(ROOT/"models/Qwen3-4B"),"--device","cuda:7","--teacher",
        "--out",str(EVAL/"teacher")],"math-teacher"))
    train=launch(["scripts/train_math_opd.py","--config","configs/math_opd.json"],"math-train")
    rc=train.wait()
    print(json.dumps({"train_exit_code":rc}),flush=True)
    if rc==0:
        jobs.append(launch(common+["--model",str(ROOT/"models/Qwen3-4B-Base"),"--device","cuda:4",
            "--adapter",str(RUN/"checkpoint-0064"),"--out",str(EVAL/"student_after")],"math-student-after"))
    codes=[p.wait() for p in jobs]
    (EVAL/"jobs.json").write_text(json.dumps({"training":rc,"evaluation_exit_codes":codes},indent=2))
    if rc or any(codes):sys.exit(1)
    results={name:json.loads((EVAL/name/"summary.json").read_text()) for name in ["student_before","teacher","student_after"]}
    results["accuracy_delta"]=results["student_after"]["accuracy"]-results["student_before"]["accuracy"]
    (EVAL/"comparison.json").write_text(json.dumps(results,indent=2))
if __name__=="__main__":main()
