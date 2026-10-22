"""Score saved responses with the same official checks as HF evaluation."""
import argparse,json,fcntl
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from eval_domains import check,summarize
def main():
 p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--out",required=True);p.add_argument("--partial",action="store_true");a=p.parse_args()
 out=Path(a.out)
 lock=(out/"grading.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX)
 if (out/"summary.json").exists():return
 rows=[json.loads(x) for x in Path(a.data).read_text().splitlines() if x.strip()]
 expected={(r["domain"],str(r["id"])):r for r in rows}
 generated=[json.loads(x) for x in (out/"generated.jsonl").read_text().splitlines() if x.strip()]
 generation=json.loads((out/"generation-config.json").read_text())
 found=[(r["domain"],str(r["id"])) for r in generated]
 complete=len(found)==len(expected) and set(found)==set(expected)
 if not a.partial and not complete:raise ValueError("Incomplete or duplicated benchmark outputs")
 results=[]
 sources=[expected[(r["domain"],str(r["id"]))] for r in generated]
 if any(r["prompt"]!=s["prompt"] for r,s in zip(generated,sources)):raise ValueError("Prompt mismatch")
 # Code is graded by subprocesses, so it runs in threads; math-verify uses signals and stays on the main thread.
 pool=ThreadPoolExecutor(32)
 pending={i:pool.submit(check,s,r["response"]) for i,(r,s) in enumerate(zip(generated,sources)) if s["domain"]=="code"}
 with (out/"cases.partial").open("w",buffering=1) as f:
  for i,(r,source) in enumerate(zip(generated,sources)):
   verdict=pending[i].result() if i in pending else check(source,r["response"])
   scored={**r,**({"benchmark":source["benchmark"]} if "benchmark" in source else {}),**verdict,"model_version_id":generation.get("model_version_id"),
    "protocol_id":generation.get("protocol_id"),"protocol_sha256":generation.get("protocol_sha256"),
    "inference_backend":generation.get("backend")};results.append(scored)
   f.write(json.dumps(scored,ensure_ascii=False)+"\n")
   if len(results)%25==0 or len(results)==len(generated):
    domains=summarize(results);benchmarks=summarize(results,"benchmark")
    for d in (*domains.values(),*benchmarks.values()):
     d["scored_count"]=d["count"]-d["grade_errors"]
     if d["grade_errors"]:d["accuracy"]=None
    progress={"scored_count":len(results),"total":len(rows),"domains":domains,**({"benchmarks":benchmarks} if benchmarks else {})}
    temp=out/"scoring-progress.tmp"
    temp.write_text(json.dumps(progress));temp.replace(out/"scoring-progress.json")
 if not complete:return
 (out/"cases.partial").replace(out/"cases.jsonl")
 result={"done":len(results),"total":len(rows),"domains":domains,**({"benchmarks":benchmarks} if benchmarks else {}),
  "scope":"All rows of recorded input; scorer per domain: math-verify (math), official strict IFEval (instruction), sandboxed test execution (code; LiveCodeBench per official harness)",
  "generation":generation}
 (out/"summary.json").write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=="__main__":main()
