"""Summarize an eval suite: per benchmark score, repeat spread, and paired bootstrap CIs.

Score per model = mean over its repeats of the primary metric (MATH, IFEval, LiveCodeBench strict;
HumanEval, MBPP lenient, as in unified-v1). Repeat spread = max - min across repeats (evaluation
noise). For each pair in --pairs, the 95% CI of the difference comes from a paired bootstrap over
problems (rep1 of each model). Extra slices: MATH500 without near-duplicates of training prompts
(manifest contamination list), LiveCodeBench by difficulty.

  python scripts/collect_eval_suite.py configs/eval_suites/v2-scale256.json --pairs mopd-mic:mix-mic mopd-mic:merge-mic
"""
import argparse, json, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
LENIENT_PRIMARY = {"humaneval164", "mbpp257"}


def cases(path):
    return {(r["domain"], str(r["id"])): r for r in map(json.loads, (path / "cases.jsonl").read_text().splitlines())}


def verdict(r):
    return bool(r["correct_lenient"] if r["benchmark"] in LENIENT_PRIMARY else r["correct"])


def slices(rows, near_dups):
    out = {}
    for key, r in rows.items():
        b = r["benchmark"]
        out.setdefault(b, []).append(key)
        if b == "math500" and r["id"] not in near_dups:
            out.setdefault("math500-decontam", []).append(key)
        if b == "livecodebench342":
            out.setdefault(f"lcb-{r['metadata']['difficulty']}" if "metadata" in r else "lcb", []).append(key)
    return out


def bootstrap(a, b, keys, n=10000, seed=0):
    diffs = [verdict(a[k]) - verdict(b[k]) for k in keys]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return 100 * sum(diffs) / len(diffs), 100 * means[int(0.025 * n)], 100 * means[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suite")
    ap.add_argument("--pairs", nargs="*", default=[])
    a = ap.parse_args()
    suite = json.loads(Path(a.suite).read_text())
    root = ROOT / "results/eval-suites" / suite["name"]
    manifest = json.loads((Path(suite["data"]).parent / "manifest.json").read_text())
    near = {i for i, _ in manifest["contamination_vs_train_prompts"]["math500"]["near_duplicates"]}
    data = {json.loads(l)["id"]: json.loads(l) for l in Path(suite["data"]).read_text().splitlines()}
    reps = {}
    for m in suite["models"]:
        found = sorted(p for p in (root / m["name"]).glob("rep*") if (p / "cases.jsonl").exists())
        if found:
            reps[m["name"]] = [cases(p) for p in found]
    for runs in reps.values():  # benchmark metadata (difficulty) comes from the data file
        for run in runs:
            for (d, i), r in run.items():
                r["metadata"] = data[i].get("metadata", {}) if i in data else {}
    first = next(iter(reps.values()))[0]
    groups = slices(first, near)
    table = {}
    for name, runs in reps.items():
        row = {}
        for g, keys in groups.items():
            scores = [100 * sum(verdict(run[k]) for k in keys) / len(keys) for run in runs]
            row[g] = {"n": len(keys), "score": round(sum(scores) / len(scores), 1),
                      **({"repeats": [round(s, 1) for s in scores], "spread": round(max(scores) - min(scores), 1)} if len(runs) > 1 else {})}
        table[name] = row
    pairs = {}
    for p in a.pairs:
        x, y = p.split(":")
        pairs[p] = {g: [round(v, 1) for v in bootstrap(reps[x][0], reps[y][0], keys)] for g, keys in groups.items()}
    result = {"suite": suite["name"], "protocol_id": suite["protocol_id"], "models": table, "paired_bootstrap_95": pairs}
    (root / "summary.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    order = list(groups)
    print("model".ljust(11) + "".join(g[:16].rjust(17) for g in order))
    for name, row in table.items():
        print(name.ljust(11) + "".join((f"{row[g]['score']:.1f}" + (f" ±{row[g]['spread']/2:.1f}" if "spread" in row[g] else "")).rjust(17) for g in order))
    for p, v in pairs.items():
        print(p.ljust(20) + "  ".join(f"{g}: {d:+.1f} [{lo:+.1f},{hi:+.1f}]" for g, (d, lo, hi) in v.items()))


if __name__ == "__main__":
    main()
