"""Regression: HTTP threads must delegate signal-based grading to a process."""
import concurrent.futures,json,os,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_math_scoring_from_http_thread():
 with tempfile.TemporaryDirectory(dir=ROOT/"monitor") as tmp:
  out=Path(tmp)
  row={"id":"regression","domain":"math","prompt":"1+1?","solution":r"\boxed{2}"}
  (out/"data.jsonl").write_text(json.dumps(row)+"\n")
  (out/"generated.jsonl").write_text(json.dumps({**row,"response":r"\boxed{2}","truncated":False})+"\n")
  (out/"generation-config.json").write_text("{}")
  def run():
   return subprocess.run([str(ROOT/"mopd_train/bin/python"),str(ROOT/"scripts/grade_benchmark_outputs.py"),"--data",str(out/"data.jsonl"),"--out",str(out),"--partial"],capture_output=True,text=True,check=True)
  with concurrent.futures.ThreadPoolExecutor(1) as pool:pool.submit(run).result()
  score=json.loads((out/"summary.json").read_text())["domains"]["math"]
  assert score["accuracy"]==1.0 and score["grade_errors"]==0,score
if __name__=="__main__":
 test_math_scoring_from_http_thread()
 print("PASS: correct math answer graded from HTTP-thread workflow")
