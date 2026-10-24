"""Result reports that used to be one-off scripts.

python scripts/report.py sweep configs/sweeps/code-full-v1.json [--protocol lcbdev-v3-greedy8192 ...]
    every variant x checkpoint x protocol: all / functional / stdin / hard / truncated and the
    difference to base under the same protocol, sorted by the first protocol's score. Protocols
    default to the sweep's eval protocol; listing several puts them side by side (e.g. old vs v3).
python scripts/report.py opd runs/scratch/mopd-bench-*      per-step medians of MOPD runs
"""

import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from mopd.evaluation import report


def sweep(a):
    from run_sweep import Sweep

    s = Sweep(a.spec)
    protocols = a.protocol or [s.spec["eval"]["protocol_id"]]
    rows = report.load_rows(s.spec["eval"]["data"])
    table, base = [], {}
    for v in s.spec["variants"]:
        for ck in s.checkpoints(v):
            cells = {}
            for p in protocols:
                out = s.eval_dir(v, ck).parent / p
                if (out / "cases.jsonl").exists():
                    cells[p] = report.lcb_breakdown(out / "cases.jsonl", rows)
            if cells:
                table.append((v["name"], ck, cells))
                if v.get("base"):
                    base = cells
    key = lambda r: -(r[2].get(protocols[0], {}).get("all") or -1)
    head = (
        "| variant | ckpt | "
        + " | ".join(f"{p}: all (Δbase) / functional / stdin / hard / trunc" for p in protocols)
        + " |"
    )
    print(head)
    print("|---|---|" + "---|" * len(protocols))
    for name, ck, cells in sorted(table, key=key):
        parts = []
        for p in protocols:
            c = cells.get(p)
            if not c:
                parts.append("")
                continue
            d = f" ({c['all'] - base[p]['all']:+.1f})" if p in base else ""
            parts.append(f"{c['all']}{d} / {c['functional']} / {c['stdin']} / {c['hard']} / {c['truncated']}")
        print(f"| {name} | {ck} | " + " | ".join(parts) + " |")


def opd(a):
    keys = [
        "steps",
        "seconds",
        "publish_seconds",
        "generation_seconds",
        "teacher_seconds",
        "student_seconds",
        "rollout_busy_fraction",
        "rollout_generated_tok_s",
        "rollout_tokens_per_step",
        "rollout_mfu",
        "rollout_mbu",
        "student_mfu_in_phase",
        "student_mfu_whole_step",
        "student_peak_gib",
    ]
    print("| run | " + " | ".join(keys) + " |")
    print("|---|" + "---|" * len(keys))
    for run in a.runs:
        f = Path(run) / "metrics.jsonl"
        if f.exists():
            r = report.opd_step_summary(f)
            print(f"| {Path(run).name} | " + " | ".join(str(r.get(k, "")) for k in keys) + " |")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sweep")
    p.add_argument("spec")
    p.add_argument("--protocol", nargs="*")
    p.set_defaults(fn=sweep)
    p = sub.add_parser("opd")
    p.add_argument("runs", nargs="+")
    p.set_defaults(fn=opd)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
