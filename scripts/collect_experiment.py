"""Flatten one experiment's evaluations (from its plan.json) into results/experiments/<name>/table.jsonl.

One row per (arm, step, benchmark) with strict and lenient accuracy; merged models are
step = final training steps (they integrate final experts). Also records each run's
provenance commit so a figure can state exactly which code produced it.
"""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument("plan");a=p.parse_args()
    plan=json.loads((ROOT/a.plan).read_text());protocol=plan["evaluation"]["protocol_id"]
    name=plan["experiment"]["name"];rows=[]
    spec=json.loads(Path(plan["experiment"]["spec_path"]).read_text());steps=spec["controls"]["steps"]
    for job in plan["jobs"]:
        if job["kind"]=="eval_checkpoint":
            run=Path(job["model_version"]).parent
            arm=run.name.removeprefix(spec["output_prefix"].rsplit("/",1)[1])
            step=int(job["checkpoint"].rsplit("-",1)[1])
            summary=run/"evaluations"/job["checkpoint"]/protocol/"summary.json"
            commit=json.loads((run/"provenance.json").read_text())["commit"]
        elif job["kind"]=="eval_model":
            arm=job["id"].split(":",1)[1];step=steps;summary=ROOT/job["out"]/"summary.json";commit=None
        else:continue
        s=json.loads(summary.read_text())
        for bench,b in s["benchmarks"].items():
            rows.append({"experiment":name,"arm":arm,"step":step,"benchmark":bench,"accuracy":b["accuracy"],
                         "accuracy_lenient":b.get("accuracy_lenient"),"count":b["count"],"commit":commit})
    out=ROOT/"results/experiments"/name;out.mkdir(parents=True,exist_ok=True)
    (out/"table.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    print(len(rows),"rows ->",out/"table.jsonl")
if __name__=="__main__":main()
