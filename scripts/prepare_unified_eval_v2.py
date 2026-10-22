"""unified-v2: MATH500 + IFEval 541 + LiveCodeBench v5/v6 (342), HumanEval/MBPP kept as diagnostics.

Why v2: MATH128 is too small (one problem = 0.8 points); HumanEval/MBPP are near saturation and
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
import gzip, hashlib, json, re
from pathlib import Path
from mopd.evaluation.livecodebench import decode_private_tests
ROOT = Path(__file__).resolve().parents[1]
VERSION = "unified-v2"
OUT = ROOT / "data/full_eval" / VERSION
LCB_INSTRUCTION = ("You will be given a question (problem specification) and will generate a correct Python program "
                   "that matches the specification and passes all tests.\n\nQuestion: {question}\n\n")
LCB_STDIN = ("Read the inputs from stdin, solve the problem and write the answer to stdout (do not directly test on "
             "the sample inputs). Reply with the complete program in a single ```python code block.")
LCB_FUNCTIONAL = ("Use the following starter code and reply with the complete solution in a single ```python code "
                  "block.\n```python\n{starter}\n```")
TRAIN_POOLS = ["data/sft/numina_v3/train.jsonl", "data/sft/smol_constraints/train.jsonl", "data/sft/opencoder/train.jsonl"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def math500():
    for r in map(json.loads, (ROOT / "data/math/upstream/math500/test.jsonl").read_text().splitlines()):
        if "\\boxed{" not in r["solution"]:
            raise ValueError(f"{r['unique_id']}: solution without \\boxed answer")
        yield {"id": r["unique_id"], "domain": "math", "split": "eval", "source": "HuggingFaceH4/MATH-500",
               "prompt": r["problem"], "problem": r["problem"], "solution": r["solution"],
               "level": r["level"], "type": r["subject"], "benchmark": "math500"}


def livecodebench(tests_dir):
    for name in ("test5.jsonl", "test6.jsonl"):
        for line in (ROOT / "data/code/upstream/livecodebench" / name).open():
            r = json.loads(line)
            tests = json.loads(r["public_test_cases"]) + decode_private_tests(r["private_test_cases"])
            kinds = {t["testtype"] for t in tests}
            if len(kinds) != 1:
                raise ValueError(f"{r['question_id']}: mixed test types {kinds}")
            kind = kinds.pop()
            path = tests_dir / f"{r['question_id']}.json.gz"
            path.write_bytes(gzip.compress(json.dumps(tests).encode(), mtime=0))
            meta = {"benchmark": "livecodebench", "testtype": kind, "tests_path": str(path.relative_to(ROOT)),
                    "tests_sha256": sha(path), "num_tests": len(tests), "difficulty": r["difficulty"],
                    "platform": r["platform"], "contest_date": r["contest_date"], "release_file": name}
            prompt = LCB_INSTRUCTION.format(question=r["question_content"])
            if kind == "functional":
                meta["func_name"] = json.loads(r["metadata"])["func_name"]
                prompt += LCB_FUNCTIONAL.format(starter=r["starter_code"])
            else:
                prompt += LCB_STDIN
            yield {"id": f"lcb/{r['question_id']}", "domain": "code", "split": "eval",
                   "source": "livecodebench/code_generation_lite", "prompt": prompt, "metadata": meta,
                   "benchmark": "livecodebench342"}


def ngrams(text, n=13):
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def overlap(rows, field, threshold=0.3):
    """Per benchmark problem, the largest share of its 13-grams found in one training prompt.

    Shared 13-grams alone are mostly boilerplate (number runs, LeetCode's "subsequence" wording);
    a near-duplicate covers a large share of the problem from a single prompt. Rows at or above
    the threshold are listed so a decontaminated score can be reported next to the full one."""
    grams = {r["id"]: ngrams(field(r)) for r in rows}
    where = {}
    for rid, gs in grams.items():
        for g in gs:
            where.setdefault(g, []).append(rid)
    best = dict.fromkeys(grams, 0.0)
    for pool in TRAIN_POOLS:
        for line in (ROOT / pool).open():
            shared = {}
            for g in ngrams(json.loads(line)["prompt"]) & where.keys():
                for rid in where[g]:
                    shared[rid] = shared.get(rid, 0) + 1
            for rid, c in shared.items():
                best[rid] = max(best[rid], c / len(grams[rid]))
    report = {"threshold": threshold}
    for bench in dict.fromkeys(r["benchmark"] for r in rows):
        ids = sorted((rid for rid in grams if rid in {r["id"] for r in rows if r["benchmark"] == bench} and best[rid] >= threshold),
                     key=lambda rid: -best[rid])
        report[bench] = {"rows": sum(r["benchmark"] == bench for r in rows), "near_duplicates": [[rid, round(best[rid], 3)] for rid in ids],
                         "max_coverage": round(max(best[r["id"]] for r in rows if r["benchmark"] == bench), 3)}
    return report


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    tests_dir = OUT / "tests/livecodebench"
    tests_dir.mkdir(parents=True)
    v1 = ROOT / "data/full_eval/unified-v1"
    reused = {b: [json.loads(l) for l in (v1 / f"{b}.jsonl").read_text().splitlines()] for b in ("ifeval541", "humaneval164", "mbpp257")}
    rows = list(math500()) + reused["ifeval541"] + list(livecodebench(tests_dir)) + reused["humaneval164"] + reused["mbpp257"]
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
    sources = [ROOT / "data/math/upstream/math500/test.jsonl", *(ROOT / "data/code/upstream/livecodebench").glob("test*.jsonl"),
               *(v1 / f"{b}.jsonl" for b in reused)]
    manifest["sources"] = {str(p): sha(p) for p in sources}
    manifest["contamination_vs_train_prompts"] = overlap(
        [r for r in rows if r["benchmark"] in ("math500", "livecodebench342")], lambda r: r.get("problem") or r["prompt"].split("Question: ", 1)[1])
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    main()
