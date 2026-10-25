"""A study's results in one table: every run x eval suite x checkpoint, accuracy per benchmark, the
difference from the baseline model and a paired McNemar test against it (the same questions, so
noise between two checkpoints shows as discordant pairs, not as a mean difference).

The baseline's results are looked up by protocol among earlier evaluations of that model
(results/**/base/<protocol_id>/ or results/baselines/<model>/<protocol_id>/).
"""

import json
import math
from collections import defaultdict
from pathlib import Path

from mopd import paths


def cases(out_dir):
    f = Path(out_dir) / "cases.jsonl"
    if not f.exists():
        return None
    return {c["id"]: c for c in map(json.loads, f.read_text().splitlines()) if c.get("id") is not None}


def by_benchmark(cs):
    acc = defaultdict(list)
    for c in cs.values():
        acc[c.get("benchmark") or c.get("domain") or "all"].append(bool(c["correct"]))
    return {b: round(100 * sum(v) / len(v), 1) for b, v in sorted(acc.items())}


def mcnemar(base, model):
    """(model right & base wrong, base right & model wrong, two-sided p) over the shared questions."""
    ids = base.keys() & model.keys()
    b = sum(model[i]["correct"] and not base[i]["correct"] for i in ids)
    c = sum(base[i]["correct"] and not model[i]["correct"] for i in ids)
    p = math.erfc(abs(b - c) / math.sqrt(2 * (b + c))) if b + c else 1.0
    return b, c, round(p, 4)


def baseline_dir(model_rel, protocol_id):
    for pattern in (f"results/baselines/{Path(model_rel).name}/{protocol_id}", f"results/**/base/{protocol_id}"):
        for d in sorted(paths.ROOT.glob(pattern)):
            if (d / "cases.jsonl").exists():
                return d
    return None


def table(study, reg, runs):
    """Rows: {run, suite, checkpoint, overall, per benchmark, delta vs baseline, mcnemar}."""
    rows = []
    model_rel = reg["models"][study.get("baseline", "qwen3-4b-base")]["path"]
    bases = {}
    for r in runs:
        for suite in r["evals"]:
            e = reg["evals"][suite]
            if suite not in bases:
                d = baseline_dir(model_rel, e["protocol_id"])
                bases[suite] = cases(d) if d else None
                if bases[suite]:
                    rows.append({"run": "baseline", "suite": suite, "checkpoint": "-", "overall": overall(bases[suite]),
                                 "benchmarks": by_benchmark(bases[suite])})
            for step in study.get("checkpoints") or [r["config"]["steps"]]:
                cs = cases(r["run_dir"] / "evaluations" / f"checkpoint-{step:04d}" / e["protocol_id"])
                if cs is None:
                    rows.append({"run": r["name"], "suite": suite, "checkpoint": step, "overall": None})
                    continue
                row = {"run": r["name"], "suite": suite, "checkpoint": step, "overall": overall(cs), "benchmarks": by_benchmark(cs)}
                if bases[suite]:
                    row["delta"] = round(row["overall"] - overall(bases[suite]), 1)
                    row["mcnemar"] = mcnemar(bases[suite], cs)
                rows.append(row)
    return rows


def overall(cs):
    return round(100 * sum(bool(c["correct"]) for c in cs.values()) / len(cs), 1)


def markdown(rows):
    """One table per eval suite; benchmarks as columns."""
    out = []
    for suite in dict.fromkeys(r["suite"] for r in rows):
        rs = [r for r in rows if r["suite"] == suite]
        benches = sorted({b for r in rs for b in r.get("benchmarks", {})})
        out += [f"### {suite}", "", "| run | ckpt | all | Δbase | +/− vs base (p) | " + " | ".join(benches) + " |",
                "|---|---|---|---|---|" + "---|" * len(benches)]
        for r in rs:
            if r["overall"] is None:
                out.append(f"| {r['run']} | {r['checkpoint']} | pending | | | " + " | ".join("" for _ in benches) + " |")
                continue
            m = r.get("mcnemar")
            sig = f"+{m[0]} −{m[1]} ({m[2]})" if m else ""
            delta = f"{r['delta']:+.1f}" if "delta" in r else ""
            out.append(f"| {r['run']} | {r['checkpoint']} | {r['overall']} | {delta} | {sig} | "
                       + " | ".join(str(r["benchmarks"].get(b, "")) for b in benches) + " |")
        out.append("")
    return "\n".join(out)
