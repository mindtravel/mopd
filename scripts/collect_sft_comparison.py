"""Collect every SFT-vs-MOPD evaluation under one protocol into a single results directory.

Copies each model's summary.json (small) and writes one flat table, so the comparison
can be read without walking ~30 run directories. cases.jsonl stays in the runs (large);
the table records its path for per-case analysis.
"""
import argparse,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def models(protocol):
    mopd="20261015-dual-smol-constraints-from-base-vllm"
    yield "base","base",ROOT/f"runs/{mopd}/evaluations/base/{protocol}",None
    yield "mopd-mi","mopd",ROOT/f"runs/{mopd}/evaluations/checkpoint-0064/{protocol}",ROOT/f"runs/{mopd}"
    for run in sorted(ROOT.glob("runs/20261021-sftv2*-lora")):
        tag=run.name.removeprefix("20261021-").removesuffix("-lora")
        variant,combo=tag.split("-")
        yield tag,variant,run/f"evaluations/checkpoint-0064/{protocol}",run
def main():
    p=argparse.ArgumentParser();p.add_argument("--protocol-id",default="unified-imc-plainqa-greedy4096-v1")
    p.add_argument("--out",default="results/sft-comparison");a=p.parse_args()
    out=ROOT/a.out;(out/"summaries").mkdir(parents=True,exist_ok=True);rows=[]
    for name,variant,ev,run in models(a.protocol_id):
        s=json.loads((ev/"summary.json").read_text());shutil.copy(ev/"summary.json",out/"summaries"/f"{name}.json")
        b=s["benchmarks"];row={"model":name,"variant":variant,
            **{k:v["accuracy"] for k,v in b.items()},"code421":s["domains"]["code"]["accuracy"],
            "ifeval_instruction_level":s["domains"]["instruction"]["strict_instruction_accuracy"],
            "truncated":sum(d["truncated"] for d in s["domains"].values()),"cases":str(ev/"cases.jsonl")}
        if run:
            config=json.loads((run/"config.json").read_text())
            row.update(seed=config.get("seed"),lr=config.get("lr"),steps=config.get("steps"),
                domains=list(config["domains"]),train_tokens=sum(json.loads(l)["tokens"] if "tokens" in json.loads(l)
                    else sum(v["tokens"] for v in json.loads(l)["domains"].values()) for l in open(run/"metrics.jsonl")))
        rows.append(row)
    (out/"table.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    (out/"README.md").write_text(f"# SFT 对比结果汇总\n\n协议：`{a.protocol_id}`（MATH128 + IFEval541 + HumanEval164 + MBPP257）。\n"
        "`table.jsonl` 每行一个模型；`summaries/` 是各评测 summary.json 的拷贝；逐题结果见 table 中 `cases` 路径。\n"
        "由 `scripts/collect_sft_comparison.py` 生成。\n")
    print(len(rows),"models ->",out)
if __name__=="__main__":main()
