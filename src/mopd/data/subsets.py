"""Training-data subsets: one canonical, decontaminated row set per source.

A subset turns one raw download into rows of a fixed schema:
  {"id", "problem_id", "source", "problem", "response", "code", "kind", "meta": {...}}
  kind: "stdin" (read stdin / print), "function" (write a named function), "class" (LeetCode
  `class Solution` starter), "unknown".  response is the full reply (explanation + code, or a
  reasoning trace); code is the final program alone.
Builders only parse and normalize. Selection (which sources, difficulty, correctness), formatting
(prompt template, code-only vs full reply) and sizes belong to recipes (mopd.data.recipes), so an
ablation never rebuilds a subset.

Build steps, recorded in data/subsets/<name>/manifest.json with counts after each step:
  read -> drop empty -> exact dedup of (problem, response) -> drop rows that near-duplicate any
  benchmark problem (mopd.data.decontam over every evaluation file in EVAL_FILES).
"""
import glob, hashlib, inspect, json, re
from pathlib import Path

from mopd.data import decontam
from mopd.data.pool import text_hash  # same whitespace normalization as exact dedup below and validate_holdout

ROOT = Path("/ssd/lzx/mopd")
EVAL_FILES = ["data/full_eval/unified-v2/all.jsonl", "data/full_eval/lcb-dev-v1/lcbdev.jsonl"]
FENCE = re.compile(r"```(?:python|py|Python)?\s*\n(.*?)```", re.DOTALL)
THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
# Reasoning-chain self-correction markers ("Wait, ... Hmm, so ... Let me think ..."). A normal,
# occasionally-backtracking chain uses a few of these; a non-converging one repeats them for
# hundreds of occurrences without reaching a conclusion (found diagnosing rstar-seed: training
# responses that pass their tests still median 67 of these in ~26k characters, and the eval-time
# failures that never produced code at all median 120 -- not a different failure mode, a denser
# version of what the training data already does a lot of). Readers that count this in `meta` let
# a recipe prefer the least hedgy of several candidate solutions to one problem, instead of a
# random one, without discarding data: mopd.data.recipes' `prefer_fewest`.
HEDGE = re.compile(r"\bWait\b|\bHmm\b|\bLet me think\b", re.I)


def last_code(text):
    blocks = FENCE.findall(text or "")
    return blocks[-1].strip("\n") if blocks else ""


def hedge_count(text):
    return len(HEDGE.findall(text or ""))


def parquet_rows(pattern, columns=None):
    import pyarrow.parquet as pq
    for f in sorted(glob.glob(str(ROOT / pattern))):
        try:
            pf = pq.ParquetFile(f)
        except Exception as e:
            # A file mid-download is unreadable (no footer yet), not corrupt data: skip it rather
            # than crash, but say so loudly -- a subset built this way is a preliminary read, not
            # the full source, and builder_version() won't catch that it changed mid-read.
            print(f"parquet_rows: skipping unreadable (likely still downloading) {f}: {e!r}")
            continue
        for batch in pf.iter_batches(batch_size=2048, columns=columns):
            yield from batch.to_pylist()


# ---------------------------------------------------------------- source readers
def read_kodcode():
    """KodCode-V1-SFT-4o: synthetic function-style tasks (incl. ones derived from Taco, Codeforces, Apps,
    Code_Contests, Leetcode), GPT-4o reply with explanation + code, verified by unit tests."""
    for r in parquet_rows("data/raw/kodcode-v1-sft-4o/data/train-*.parquet",
                          columns=["conversations", "subset", "question", "4o_solution", "question_id",
                                   "style", "4o_correctness", "gpt_difficulty", "gpt_pass_percentage"]):
        reply = next((c["value"] for c in r["conversations"] if c["from"] == "gpt"), "")
        yield {"id": f"kodcode/{r['question_id']}", "problem_id": r["question_id"], "source": r["subset"],
               "problem": r["question"], "response": reply, "code": last_code(reply) or r["4o_solution"],
               "kind": "function",
               "meta": {"style": r["style"], "correct": r["4o_correctness"] == "True", "gpt_difficulty": r["gpt_difficulty"],
                        "pass_rate": r["gpt_pass_percentage"]}}


def read_opencodereasoning():
    """OpenCodeReasoning split_0: competitive programming (CodeContests, TACO, APPS, Codeforces) with a
    DeepSeek-R1 reply (reasoning + code) and the code alone; several replies per problem, not verified.
    Upstream `id` is shared across a problem's sampled completions, not unique per row (measured up
    to 18 rows sharing one id), so it cannot be the row id; a running counter plus a text digest is."""
    for i, r in enumerate(parquet_rows("data/raw/opencodereasoning/split_0/*.parquet",
                                      columns=["output", "input", "solution", "dataset", "difficulty", "id"])):
        reply = THINK.sub("", r["output"]) if "</think>" in r["output"] else r["output"]
        kind = "stdin" if re.search(r"input\(\)|sys\.stdin", r["solution"]) else "function" if "def " in r["solution"] else "unknown"
        # problem_id must key off the same normalized text as the exact-dedup hash below and
        # mopd.data.pool.validate_holdout, or two whitespace-variant copies of one problem get
        # treated as different problems: one can then land in train and the other in dev, which
        # validate_holdout then (correctly) rejects as a literal prompt collision.
        yield {"id": f"ocr/{i}-{text_hash(r['input'] + chr(0) + r['output'])[:12]}", "problem_id": text_hash(r["input"])[:16],
               "source": r["dataset"], "problem": r["input"], "response": r["output"],
               "code": last_code(r["solution"]) or r["solution"], "kind": kind,
               "meta": {"difficulty": r["difficulty"], "trace_chars": len(r["output"]), "final_chars": len(reply), "upstream_id": r["id"]}}


def read_rstar_seed():
    """rStar-Coder seed_sft: QwQ-32B reasoning+code solutions for real competitive-programming
    problems (AIZU/AtCoder/CodeChef/Codeforces/USACO), majority-verified against generated tests
    (microsoft/rStar-Coder, arXiv:2505.21297). `verified`/`is_passed` select the usable rows (recipe
    `where`, like KodCode's `correct`). QwQ shows the same failure as OpenCodeReasoning's R1 teacher
    -- switching to Chinese/Cyrillic mid-response -- at a far lower rate (5.3% of verified+passed
    rows here vs 0.1% of all OpenCodeReasoning rows, which still blew up to 66% of a trained model's
    outputs); `clean_language` flags it so recipes can filter it out rather than learn it again.
    `hedge_markers` (mopd.data.subsets.hedge_count) lets a recipe prefer the least rambling of a
    problem's several passing solutions instead of a random one (most problems have several: median
    15 candidates, 90% have >=3) -- diagnosed separately: trained models reproduce this teacher's
    heavy self-correction style and, on problems they cannot actually solve, loop in it forever
    instead of concluding; training responses that DO pass already median 67 of these markers.
    Upstream `question_id` is shared across a problem's sampled solutions too (up to 16 rows/id,
    same issue as OpenCodeReasoning's `id`), so it is only used to break ties; problem_id itself is
    a text digest -- two different question_ids for byte-identical questions (observed: duplicate
    classic problems re-added under a new id) must still land in the same problem_id group, or
    per_problem=1 fails to dedupe them and recipes.build() can hand out the same rendered prompt
    twice, which mopd.data.pool.build_plan then (correctly) rejects as a training-pool collision."""
    nonlatin = re.compile(r"[Ѐ-ӿ一-鿿]")
    for i, r in enumerate(parquet_rows("data/raw/rstar-coder/seed_sft/*.parquet")):
        kind = "function" if r["starter_code"].strip() else "stdin"
        yield {"id": f"rstar/{i}-{text_hash(r['question'] + chr(0) + r['response'])[:12]}",
               "problem_id": text_hash(r["question"])[:16], "source": "rstar-coder-seed",
               "problem": r["question"], "response": r["response"], "code": r["code"] or last_code(r["response"]),
               "kind": kind,
               "meta": {"verified": bool(r["verified"]), "is_passed": bool(r["is_passed"]),
                        "clean_language": not bool(nonlatin.search(r["response"])),
                        "hedge_markers": hedge_count(r["response"])}}


def read_opencoder_clean():
    """The current code pool (OpenCoder stage 2 educational + evol, opener boilerplate removed)."""
    for line in (ROOT / "data/sft/opencoder_stage2_clean/train.jsonl").open():
        r = json.loads(line)
        yield {"id": f"opencoder/{r['id']}", "problem_id": str(r["id"]), "source": r.get("source", "opencoder"),
               "problem": r["prompt"], "response": r["response"], "code": last_code(r["response"]), "kind": "function",
               "meta": {}}


SUBSETS = {
    "kodcode-4o": {"reader": read_kodcode, "raw": "data/raw/kodcode-v1-sft-4o"},
    "opencodereasoning": {"reader": read_opencodereasoning, "raw": "data/raw/opencodereasoning"},
    "opencoder-clean": {"reader": read_opencoder_clean, "raw": "data/sft/opencoder_stage2_clean"},
    "rstar-seed": {"reader": read_rstar_seed, "raw": "data/raw/rstar-coder"},
}


# ---------------------------------------------------------------- build
def builder_version(name):
    """Digest of this subset's reader, the shared build steps, the decontamination code and the raw
    download manifest: changing any of them yields a new version; other subsets are unaffected."""
    h = hashlib.sha256()
    for part in (SUBSETS[name]["reader"], build, last_code):
        h.update(inspect.getsource(part).encode())
    h.update(Path(decontam.__file__).read_bytes() + json.dumps(EVAL_FILES).encode())
    raw = ROOT / SUBSETS[name]["raw"] / "download-manifest.json"
    h.update(raw.read_bytes() if raw.exists() else SUBSETS[name]["raw"].encode())
    return h.hexdigest()[:12]


def eval_index():
    problems = {}
    for f in EVAL_FILES:
        for line in (ROOT / f).open():
            r = json.loads(line)
            problems[f"{r['benchmark']}:{r['id']}"] = r.get("problem") or r["prompt"].split("Question: ", 1)[-1]
    return decontam.BenchmarkIndex(problems)


def build(name, out_root=ROOT / "data/subsets"):
    out = out_root / name
    version = builder_version(name)
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["version"] == version:
        return json.loads(manifest_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    index = eval_index()
    counts = {"read": 0, "empty": 0, "duplicate": 0, "contaminated": 0, "kept": 0}
    seen, dropped, verdict = set(), [], {}
    tmp = out / "rows.jsonl.tmp"
    with tmp.open("w") as f:
        for r in SUBSETS[name]["reader"]():
            counts["read"] += 1
            if not r["problem"].strip() or not (r["response"].strip() or r["code"].strip()):
                counts["empty"] += 1; continue
            key = hashlib.sha1((" ".join(r["problem"].split()) + "\0" + r["response"]).encode()).digest()
            if key in seen:
                counts["duplicate"] += 1; continue
            seen.add(key)
            pkey = hashlib.sha1(r["problem"].encode()).digest()  # several replies share one problem
            if pkey not in verdict:
                verdict[pkey] = index.worst(r["problem"])
            pid, cov = verdict[pkey]
            if cov >= decontam.THRESHOLD:
                counts["contaminated"] += 1; dropped.append([r["id"], pid, round(cov, 3)]); continue
            counts["kept"] += 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(out / "rows.jsonl")
    sha = hashlib.sha256((out / "rows.jsonl").read_bytes()).hexdigest()
    manifest = {"name": name, "version": version, "rows_sha256": sha, "counts": counts,
                "decontam": {"threshold": decontam.THRESHOLD, "eval_files": EVAL_FILES, "dropped": dropped[:2000]},
                "raw": SUBSETS[name]["raw"]}
    manifest_path.write_text(json.dumps(manifest, indent=1))
    return manifest


def rows(name, root=ROOT / "data/subsets"):
    manifest = json.loads((root / name / "manifest.json").read_text())
    if manifest["version"] != builder_version(name):
        raise ValueError(f"subset {name} is stale: rebuild it (scripts/subsets.py build {name})")
    with (root / name / "rows.jsonl").open() as f:
        for line in f:
            yield json.loads(line)
