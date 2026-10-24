"""unified-v3: MATH500 + IFEval 541 + LiveCodeBench v5/v6 (342), HumanEval/MBPP kept as diagnostics.

v3 = v2 with the LiveCodeBench rows rendered by the official prompt (livecodebench.prompt); math,
IFEval, HumanEval and MBPP rows unchanged. Why v2: MATH128 is too small (one problem = 0.8 points); HumanEval/MBPP are near saturation and
likely contaminated (Base HumanEval 84.8). LiveCodeBench test5 + test6 hold contests from
2024-09-22 to 2025-04-06, after most of the base model's data.

Sources (downloaded via hf-mirror, digests in manifest.json):
  data/math/upstream/math500/test.jsonl                     HuggingFaceH4/MATH-500
  data/code/upstream/livecodebench/test{5,6}.jsonl           livecodebench/code_generation_lite
  data/full_eval/unified-v1/{ifeval541,humaneval164,mbpp257}.jsonl   rows reused unchanged

Like v1, every sub-benchmark file holds the same JSON lines as all.jsonl. LiveCodeBench tests
(public + decoded private) go to tests/livecodebench/<question_id>.json.gz, referenced by
path and sha256 from each row. The manifest lists benchmark problems whose 13-grams are largely
covered by a single training-pool prompt (near-duplicates).
"""

import hashlib, json
from pathlib import Path
from mopd.data import decontam
from mopd.evaluation import livecodebench as lcb

ROOT = Path(__file__).resolve().parents[1]
VERSION = "unified-v3"
OUT = ROOT / "data/full_eval" / VERSION
TRAIN_POOLS = [
    "data/sft/numina_v3/train.jsonl",
    "data/sft/smol_constraints/train.jsonl",
    "data/sft/opencoder_stage2_clean/train.jsonl",
]
# must match the pools production experts actually train on (configs/experiments/*/expert-c.json);
# this listed the retired stage1 opencoder pool until caught in an architecture review (2026-10-23)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def math500():
    for r in map(json.loads, (ROOT / "data/math/upstream/math500/test.jsonl").read_text().splitlines()):
        if "\\boxed{" not in r["solution"]:
            raise ValueError(f"{r['unique_id']}: solution without \\boxed answer")
        yield {
            "id": r["unique_id"],
            "domain": "math",
            "split": "eval",
            "source": "HuggingFaceH4/MATH-500",
            "prompt": r["problem"],
            "problem": r["problem"],
            "solution": r["solution"],
            "level": r["level"],
            "type": r["subject"],
            "benchmark": "math500",
        }


def livecodebench(tests_dir):
    files = [ROOT / "data/code/upstream/livecodebench" / n for n in ("test5.jsonl", "test6.jsonl")]
    return lcb.benchmark_rows(files, tests_dir, ROOT, "livecodebench342")


def overlap(rows, field, threshold=decontam.THRESHOLD):
    """Per benchmark, problems whose 13-grams are largely covered by one training prompt."""
    index = decontam.BenchmarkIndex({r["id"]: field(r) for r in rows})
    best = index.max_coverage(json.loads(line)["prompt"] for pool in TRAIN_POOLS for line in (ROOT / pool).open())
    report = {"threshold": threshold}
    for bench in dict.fromkeys(r["benchmark"] for r in rows):
        ids = [r["id"] for r in rows if r["benchmark"] == bench]
        dups = sorted((i for i in ids if best[i] >= threshold), key=lambda i: -best[i])
        report[bench] = {
            "rows": len(ids),
            "near_duplicates": [[i, round(best[i], 3)] for i in dups],
            "max_coverage": round(max(best[i] for i in ids), 3),
        }
    return report


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    tests_dir = OUT / "tests/livecodebench"
    tests_dir.mkdir(parents=True)
    v1 = ROOT / "data/full_eval/unified-v1"
    reused = {
        b: [json.loads(l) for l in (v1 / f"{b}.jsonl").read_text().splitlines()]
        for b in ("ifeval541", "humaneval164", "mbpp257")
    }
    rows = (
        list(math500())
        + reused["ifeval541"]
        + list(livecodebench(tests_dir))
        + reused["humaneval164"]
        + reused["mbpp257"]
    )
    keys = [(r["domain"], str(r["id"])) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate benchmark ID across sources")
    lines = [(r["benchmark"], json.dumps(r, ensure_ascii=False) + "\n") for r in rows]
    files = {"all": [l for _, l in lines]}
    for name in dict.fromkeys(b for b, _ in lines):
        files[name] = [l for b, l in lines if b == name]
    manifest = {"version": VERSION, "files": {}}
    for name, body in files.items():
        path = OUT / f"{name}.jsonl"
        path.write_text("".join(body))
        manifest["files"][name] = {"path": str(path), "rows": len(body), "sha256": sha(path)}
    sources = [
        ROOT / "data/math/upstream/math500/test.jsonl",
        *(ROOT / "data/code/upstream/livecodebench").glob("test*.jsonl"),
        *(v1 / f"{b}.jsonl" for b in reused),
    ]
    manifest["sources"] = {str(p): sha(p) for p in sources}
    manifest["contamination_vs_train_prompts"] = overlap(
        [r for r in rows if r["benchmark"] in ("math500", "livecodebench342")],
        lambda r: r["problem"],
    )
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    main()
