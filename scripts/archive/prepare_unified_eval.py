"""One instruction + math + code benchmark set, with line-identical sub-benchmark files.

all.jsonl = existing data/full_eval/all.jsonl (MATH 128 + IFEval 541, unchanged rows)
          + data/code/benchmark/all.jsonl (HumanEval 164 + MBPP-sanitized test 257),
each row tagged with a top-level "benchmark" so grading reports per benchmark, not only
per domain (HumanEval and MBPP are both domain "code").

Every sub-benchmark file holds the exact same JSON lines as all.jsonl. A protocol's
hash covers prompt mode and generation settings only, not the data file
(mopd.evaluation.generation_manifest), so a sub-benchmark generated later under the same
protocol_id is the same measurement as that benchmark's slice of a full run and can
fill in a missing piece without regenerating the rest.
"""
import hashlib,json
from pathlib import Path
from mopd.data.pool import load_pool
ROOT=Path(__file__).resolve().parents[1]
VERSION="unified-v1"
def tag(row):
    if row["domain"]=="math":return "math128"
    if row["domain"]=="instruction":return "ifeval541"
    if row["domain"]=="code":return {"humaneval":"humaneval164","mbpp":"mbpp257"}[row["metadata"]["benchmark"]]
    raise ValueError("Unknown domain: "+row["domain"])
def main():
    rows=load_pool(ROOT/"data/full_eval/all.jsonl")+load_pool(ROOT/"data/code/benchmark/all.jsonl")
    keys=[(r["domain"],str(r["id"])) for r in rows]
    if len(set(keys))!=len(keys):raise ValueError("Duplicate benchmark ID across sources")
    lines=[(tag(r),json.dumps({**r,"benchmark":tag(r)},ensure_ascii=False)+"\n") for r in rows]
    out=ROOT/"data/full_eval"/VERSION;out.mkdir(parents=True,exist_ok=False)
    files={"all":[l for _,l in lines]}
    for name in dict.fromkeys(b for b,_ in lines):files[name]=[l for b,l in lines if b==name]
    manifest={"version":VERSION,"files":{}}
    for name,body in files.items():
        path=out/f"{name}.jsonl";path.write_text("".join(body))
        manifest["files"][name]={"path":str(path),"rows":len(body),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest["sources"]={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT/"data/full_eval/all.jsonl",ROOT/"data/code/benchmark/all.jsonl")}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))
if __name__=="__main__":main()
