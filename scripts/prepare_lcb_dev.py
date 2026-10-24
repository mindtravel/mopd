"""LiveCodeBench dev set for data ablations: releases v1-v4 (test.jsonl .. test4.jsonl).

unified-v2 holds v5 + v6 (2024-09-22 .. 2025-04-06) as the test set. Data choices, mixing ratios
and checkpoints are selected on this dev set only; the test set is evaluated once per finalist.
Same row format, prompt and grader as the test set (mopd.evaluation.livecodebench).

  python scripts/prepare_lcb_dev.py                # -> data/full_eval/lcb-dev-v1/{lcbdev.jsonl,tests/,manifest.json}
  python scripts/prepare_lcb_dev.py --style think  # -> data/full_eval/lcb-dev-v2-think (prompt_think, same rows/tests)
"""

import argparse, hashlib, json
from pathlib import Path
from mopd.evaluation import livecodebench as lcb

ROOT = Path(__file__).resolve().parents[1]
OUTS = {"code": ROOT / "data/full_eval/lcb-dev-v1", "think": ROOT / "data/full_eval/lcb-dev-v2-think"}
RAW = ROOT / "data/raw/livecodebench"
FILES = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"]
TEST_SET = ROOT / "data/full_eval/unified-v2/livecodebench342.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", choices=sorted(OUTS), default="code")
    a = ap.parse_args()
    OUT = OUTS[a.style]
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "tests").mkdir()
    rows = list(lcb.benchmark_rows([RAW / f for f in FILES], OUT / "tests", ROOT, "lcbdev", style=a.style))
    test_ids = {json.loads(l)["id"] for l in TEST_SET.open()}
    clash = [r["id"] for r in rows if r["id"] in test_ids]
    if clash:
        raise ValueError(f"dev rows also in the test set: {clash[:5]}")
    path = OUT / "lcbdev.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    count = lambda key: {
        v: sum(r["metadata"][key] == v for r in rows) for v in sorted({r["metadata"][key] for r in rows})
    }
    manifest = {
        "version": OUT.name,
        "prompt_style": a.style,
        "rows": len(rows),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "dates": [min(r["metadata"]["contest_date"] for r in rows), max(r["metadata"]["contest_date"] for r in rows)],
        "difficulty": count("difficulty"),
        "platform": count("platform"),
        "testtype": count("testtype"),
        "raw": json.loads((RAW / "download-manifest.json").read_text())["commit"],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
