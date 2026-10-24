"""Re-score existing generations with the current scorers, without regenerating.

Writes <eval>/grading-v2/{cases.jsonl,summary.json} next to the original grading, which
stays untouched, so strict numbers can be checked unchanged and the *_lenient scores
added by scripts/eval_domains.check become available for every past evaluation.
Code checks execute in subprocesses, so rows are scored in a process pool.
"""

import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_domains import check, summarize

SOURCE = None


def score(r):
    source = SOURCE[(r["domain"], str(r["id"]))]
    if r["prompt"] != source["prompt"]:
        raise ValueError("Prompt mismatch")
    return {
        **r,
        **({"benchmark": source["benchmark"]} if "benchmark" in source else {}),
        **check(source, r["response"]),
    }


def random_checker_rows(rows):
    bad = lambda k: not (k.get("letter") and len(k["letter"]) == 1 and "a" <= k["letter"].lower() <= "z")
    return {
        (r["domain"], str(r["id"]))
        for r in rows
        if r["domain"] == "instruction"
        and any(
            i == "keywords:letter_frequency" and bad(k)
            for i, k in zip(r["metadata"]["instruction_id_list"], r["metadata"]["kwargs"])
        )
    }


def load_rows(data):
    return [json.loads(x) for x in Path(data).read_text().splitlines() if x.strip()]


def init(data):
    global SOURCE
    SOURCE = {(r["domain"], str(r["id"])): r for r in load_rows(data)}


def regrade(ev, data, pool):
    generated = [json.loads(x) for x in (ev / "generated.jsonl").read_text().splitlines() if x.strip()]
    results = pool.map(score, generated, chunksize=8)
    out = ev / "grading-v2"
    out.mkdir(exist_ok=True)
    (out / "cases.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))
    domains = summarize(results)
    benchmarks = summarize(results, "benchmark")
    old = json.loads((ev / "summary.json").read_text())
    # Strict verdicts must reproduce the original grading, except rows whose official checker was
    # random (letter_frequency with a non a-z letter), which check() now seeds per row.
    original = {(r["domain"], str(r["id"])): r["correct"] for r in map(json.loads, open(ev / "cases.jsonl"))}
    changed = [k for k, r in ((("%s" % r["domain"], str(r["id"])), r) for r in results) if r["correct"] != original[k]]
    unexplained = [k for k in changed if k not in RANDOM_CHECKER_ROWS]
    if unexplained:
        raise ValueError(f"{ev}: strict verdict changed for {unexplained[:5]}")
    summary = {
        **{k: v for k, v in old.items() if k not in ("domains", "benchmarks")},
        "domains": domains,
        **({"benchmarks": benchmarks} if benchmarks else {}),
        "regraded_from": str(ev / "generated.jsonl"),
        "strict_changed_rows": [list(k) for k in changed],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return {
        k: (round(v["accuracy"], 3), round(v.get("accuracy_lenient", float("nan")), 3)) for k, v in benchmarks.items()
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("evals", nargs="+")
    a = p.parse_args()
    global RANDOM_CHECKER_ROWS
    RANDOM_CHECKER_ROWS = random_checker_rows(load_rows(a.data))
    print("rows with a random official checker:", sorted(RANDOM_CHECKER_ROWS), flush=True)
    with Pool(a.workers, initializer=init, initargs=(a.data,)) as pool:
        for ev in map(Path, a.evals):
            print(ev.parent.parent.parent.name, regrade(ev, a.data, pool), flush=True)


if __name__ == "__main__":
    main()
