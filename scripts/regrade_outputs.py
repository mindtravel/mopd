"""Re-score existing generations with the current scorers, without regenerating.

Writes <eval>/grading-v2/{cases.jsonl,summary.json} next to the original grading, which
stays untouched, so strict numbers can be checked unchanged and the *_lenient scores
added by scripts/eval_domains.check become available for every past evaluation.
Code checks execute in subprocesses, so rows are scored in a process pool.
"""
import argparse,json,sys
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from eval_domains import check,summarize
SOURCE=None
def score(r):
    source=SOURCE[(r["domain"],str(r["id"]))]
    if r["prompt"]!=source["prompt"]:raise ValueError("Prompt mismatch")
    return {**r,**({"benchmark":source["benchmark"]} if "benchmark" in source else {}),**check(source,r["response"])}
def init(data):
    global SOURCE
    SOURCE={(r["domain"],str(r["id"])):r for r in (json.loads(x) for x in Path(data).read_text().splitlines() if x.strip())}
def regrade(ev,data,pool):
    generated=[json.loads(x) for x in (ev/"generated.jsonl").read_text().splitlines() if x.strip()]
    results=pool.map(score,generated,chunksize=8)
    out=ev/"grading-v2";out.mkdir(exist_ok=True)
    (out/"cases.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in results))
    domains=summarize(results);benchmarks=summarize(results,"benchmark")
    old=json.loads((ev/"summary.json").read_text())
    for name,d in domains.items():  # strict scores must reproduce the original grading exactly
        if d["correct"]!=old["domains"][name]["correct"]:
            raise ValueError(f"{ev}: strict {name} correct {d['correct']} != original {old['domains'][name]['correct']}")
    summary={**{k:v for k,v in old.items() if k not in ("domains","benchmarks")},"domains":domains,
             **({"benchmarks":benchmarks} if benchmarks else {}),"regraded_from":str(ev/"generated.jsonl")}
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    return {k:(round(v["accuracy"],3),round(v.get("accuracy_lenient",float("nan")),3)) for k,v in benchmarks.items()}
def main():
    p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--workers",type=int,default=48)
    p.add_argument("evals",nargs="+");a=p.parse_args()
    with Pool(a.workers,initializer=init,initargs=(a.data,)) as pool:
        for ev in map(Path,a.evals):print(ev.parent.parent.parent.name,regrade(ev,a.data,pool),flush=True)
if __name__=="__main__":main()
