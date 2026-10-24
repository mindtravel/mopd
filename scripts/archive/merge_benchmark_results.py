"""Merge graded benchmark slices of one model version into one full-benchmark result.

Use case: a model was evaluated on part of data/full_eval/unified-v1 (e.g. all669 rows
before code existed, or one sub-benchmark re-run) and the rest was generated later.
Slices merge only if they are the same measurement: identical protocol_sha256 (prompt
mode + generation settings) and identical weights (base model + checkpoint sha256).
Rows must not overlap and their union must be exactly the target benchmark file;
each row's prompt must match the target file. Grading is not redone: each slice's
cases.jsonl was already scored by scripts/grade_benchmark_outputs.py.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from eval_domains import summarize
def identity(generation):
    return {"protocol_sha256":generation["protocol_sha256"],"base_model":generation["model"]["base_model"],
            "checkpoint_sha256":generation["model"].get("checkpoint_sha256")}
def main():
    p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--out",required=True)
    p.add_argument("parts",nargs="+");a=p.parse_args()
    expected={(r["domain"],str(r["id"])):r for r in (json.loads(x) for x in Path(a.data).read_text().splitlines() if x.strip())}
    merged={};generations=[]
    for part in map(Path,a.parts):
        generation=json.loads((part/"generation-config.json").read_text())
        if generations and identity(generation)!=identity(generations[0]):
            raise ValueError(f"{part}: different protocol or weights than {a.parts[0]}: {identity(generation)}")
        generations.append(generation)
        for line in (part/"cases.jsonl").read_text().splitlines():
            if not line.strip():continue
            r=json.loads(line);k=(r["domain"],str(r["id"]))
            if k in merged:raise ValueError(f"row {k} appears in more than one slice")
            if k not in expected:raise ValueError(f"row {k} from {part} is not in {a.data}")
            if r["prompt"]!=expected[k]["prompt"]:raise ValueError(f"prompt mismatch for {k}")
            if "benchmark" in expected[k]:r["benchmark"]=expected[k]["benchmark"]
            merged[k]=r
    missing=set(expected)-set(merged)
    if missing:raise ValueError(f"{len(missing)} rows of {a.data} not covered, e.g. {sorted(missing)[:3]}")
    results=[merged[k] for k in expected]
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    (out/"cases.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in results))
    domains=summarize(results);benchmarks=summarize(results,"benchmark")
    for d in (*domains.values(),*benchmarks.values()):
        d["scored_count"]=d["count"]-d["grade_errors"]
        if d["grade_errors"]:d["accuracy"]=None
    summary={"done":len(results),"total":len(expected),"domains":domains,**({"benchmarks":benchmarks} if benchmarks else {}),
        "scope":"Merged from graded slices with identical protocol and weights",
        "merged_from":[str(Path(x).resolve()) for x in a.parts],"generation":{**generations[0],"data_path":str(Path(a.data).resolve())}}
    (out/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary["domains"]))
if __name__=="__main__":main()
