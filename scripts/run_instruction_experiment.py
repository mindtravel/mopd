"""Run an experiment, then evaluate its final adapter with existing tools."""
import argparse,json,os,subprocess,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--eval-name",required=True);a=p.parse_args()
 c=json.loads(Path(a.config).read_text());run=Path(c["output"]);state=ROOT/"logs"/(run.name+"-supervisor.json")
 def report(**kw):
  temp=state.with_suffix(".partial");temp.write_text(json.dumps({"time":time.time(),**kw},indent=2));temp.replace(state)
 def job(args,phase):
  report(state=phase)
  subprocess.run(args,cwd=ROOT,check=True)
 try:
  job([str(ROOT/"mopd_train/bin/python"),"-u","scripts/train_multidomain_opd.py","--config",a.config],"training")
  status=json.loads((run/"status.json").read_text())
  if status["state"]!="complete":raise RuntimeError("Training not complete")
  output=ROOT/"runs/full-eval"/a.eval_name
  job([str(ROOT/"mopd_infer/bin/python"),"-u","scripts/generate_benchmark_vllm.py","--model",c["student"],"--adapter",status["final_adapter"],"--data",str(ROOT/"data/instruction/ifeval.jsonl"),"--out",str(output),"--device","3","--max-new-tokens","8192","--max-seqs","32","--batch-size","16"],"evaluating_ifeval")
  job([str(ROOT/"mopd_train/bin/python"),"-u","scripts/grade_benchmark_outputs.py","--data",str(ROOT/"data/instruction/ifeval.jsonl"),"--out",str(output)],"grading_ifeval")
  report(state="complete",evaluation=str(output))
 except BaseException:
  report(state="failed",traceback=traceback.format_exc());raise
if __name__=="__main__":main()
