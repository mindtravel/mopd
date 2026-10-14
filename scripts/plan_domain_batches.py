"""Materialize the exact prompt plan without loading models."""
import argparse,json,hashlib
from pathlib import Path
from mopd.data.pool import load_pool,build_plan,validate_holdout

def make_plan(c):
    rows=load_pool(c["train_file"])
    if any(r["domain"] != c["domain"] for r in rows):
        raise ValueError("Single-teacher run requires a single configured domain")
    for path in c.get("holdout_files",[]):
        validate_holdout(rows,load_pool(path))
    plan=build_plan(rows,seed=c["seed"],steps=c["steps"],batch_size=c["accumulation"])
    return plan

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();c=json.loads(Path(a.config).read_text());plan=make_plan(c)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x") as f:
        for step,rows in enumerate(plan,1):
            f.write(json.dumps({"step":step,"rows":rows},ensure_ascii=False)+"\n")
    print(json.dumps({"steps":len(plan),"prompts":sum(map(len,plan)),
      "sha256":hashlib.sha256(out.read_bytes()).hexdigest()}))
if __name__=="__main__":main()
