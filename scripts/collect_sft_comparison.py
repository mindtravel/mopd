"""Collect every expert-integration evaluation under one protocol into one results directory.

Rows: base, MOPD, every SFT run (all variants and seeds) and every merged model under
results/merge-evals. Each row carries strict and lenient (format-tolerant) accuracy per
benchmark; grading-v2 (scripts/regrade_outputs.py) is preferred when present because
only it has the lenient scores. Summaries are copied; cases.jsonl stays in place and
its path is recorded for per-case analysis.
"""
import argparse,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MOPD_RUNS=("20261015-dual-smol-constraints-from-base-vllm","20261021-mopd-mi-sftexpert-vllm")
def teacher_kind(run):
    """official (external post-trained teachers) vs sftexpert, from the teachers' provenance."""
    from mopd.evaluation.versioning import _teacher_source
    kinds={_teacher_source(spec["teacher"])["kind"] for spec in json.loads((run/"config.json").read_text())["domains"].values()}
    if len(kinds)!=1:raise ValueError(f"{run}: mixed teacher sources {kinds}")
    return {"external":"official","sft_expert":"sftexpert"}[kinds.pop()]
def models(protocol):
    yield "base","base",None,ROOT/f"runs/{MOPD_RUNS[0]}/evaluations/base/{protocol}",None
    for name in MOPD_RUNS:
        run=ROOT/"runs"/name
        if not (run/"config.json").exists():continue
        combo="".join(d[0] for d in json.loads((run/"config.json").read_text())["domains"])
        variant=f"mopd-{teacher_kind(run)}"
        yield f"{variant}-{combo}",variant,combo,run/f"evaluations/checkpoint-0064/{protocol}",run
    for run in sorted(ROOT.glob("runs/20261021-sftv2*-lora")):
        variant,combo=run.name.removeprefix("20261021-").removesuffix("-lora").split("-")
        yield f"{variant}-{combo}",variant,combo,run/f"evaluations/checkpoint-0064/{protocol}",run
    for ev in sorted(ROOT.glob(f"results/merge-evals/*/{protocol}")):
        name=ev.parent.name  # e.g. mic-avg
        yield f"merge-{name}","merge-"+name.split("-",1)[1],name.split("-",1)[0],ev,None
def teacher_references(protocol):
    """Official teachers on their own chat template: a separate protocol, reported as reference only."""
    for ev in sorted(ROOT.glob(f"results/teacher-evals/*/{protocol}")):
        yield f"teacher-{ev.parent.name}","teacher-official",None,ev,None
def scores(ev):
    graded=ev/"grading-v2" if (ev/"grading-v2/summary.json").exists() else ev
    s=json.loads((graded/"summary.json").read_text())
    row={"graded":str(graded),"cases":str(graded/"cases.jsonl")}
    for name,b in s["benchmarks"].items():
        row[name]=b["accuracy"];row[name+"_lenient"]=b.get("accuracy_lenient")
    inst=s["domains"]["instruction"]
    row["ifeval_instruction_level"]=inst["strict_instruction_accuracy"]
    row["ifeval_instruction_level_lenient"]=inst.get("lenient_instruction_accuracy")
    row["truncated"]=sum(d["truncated"] for d in s["domains"].values())
    return s,graded,row
def main():
    p=argparse.ArgumentParser();p.add_argument("--protocol-id",default="unified-imc-plainqa-greedy4096-v1")
    p.add_argument("--teacher-protocol-id",default="unified-imc-chat-greedy4096-v1")
    p.add_argument("--out",default="results/sft-comparison");a=p.parse_args()
    out=ROOT/a.out;(out/"summaries").mkdir(parents=True,exist_ok=True);rows=[]
    sources=[(m,a.protocol_id) for m in models(a.protocol_id)]+[(m,a.teacher_protocol_id) for m in teacher_references(a.teacher_protocol_id)]
    for (name,variant,combo,ev,run),protocol in sources:
        if not (ev/"summary.json").exists():print("skip (not graded yet):",name);continue
        s,graded,row=scores(ev);shutil.copy(graded/"summary.json",out/"summaries"/f"{name}.json")
        row={"model":name,"variant":variant,"combo":combo,"protocol":protocol,**row}
        if run:
            config=json.loads((run/"config.json").read_text())
            row.update(seed=config.get("seed"),lr=config.get("lr"),steps=config.get("steps"),domains=list(config["domains"]),
                train_tokens=sum(m["tokens"] if "tokens" in m else sum(v["tokens"] for v in m["domains"].values())
                                 for m in map(json.loads,open(run/"metrics.jsonl"))))
        rows.append(row)
    (out/"table.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    (out/"README.md").write_text(f"# 专家整合对比结果汇总\n\n协议：`{a.protocol_id}`（MATH128 + IFEval541 + HumanEval164 + MBPP257）。\n"
        "`table.jsonl` 每行一个模型，含严格分与宽松分（`*_lenient`，格式容错）与 `protocol`；官方教师行用自身 chat 模板（另一协议），只作参考；`summaries/` 是对应 summary.json 拷贝；"
        "逐题结果见 `cases` 路径。由 `scripts/collect_sft_comparison.py` 生成。\n")
    print(len(rows),"models ->",out)
if __name__=="__main__":main()
