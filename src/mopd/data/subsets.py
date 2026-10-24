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

import glob, hashlib, inspect, json, re, textwrap
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
NONLATIN = re.compile(r"[Ѐ-ӿ一-鿿]")  # Cyrillic/CJK, used by read_rstar_seed's clean_language flag


def last_code(text):
    blocks = FENCE.findall(text or "")
    return blocks[-1].strip("\n") if blocks else ""


def hedge_count(text):
    return len(HEDGE.findall(text or ""))


def _header_names():
    """Names the LiveCodeBench harness makes available before running a solution (its HEADER does
    `from math import *` etc.), so code may use them without importing."""
    from mopd.evaluation.livecodebench import HEADER

    ns = {}
    exec(HEADER, ns)  # imports only
    return set(ns)


_KNOWN = None


def undefined_names(code):
    """Names a program reads but never binds, imports or gets from builtins / the eval header, e.g. a
    call to a helper it forgot to write (`divisors(n)` in a full-FT OpenCodeInstruct student, which
    learned it from training answers like it). Flow-insensitive: a name bound anywhere counts as
    bound. Unparseable code returns {"<syntax error>"}."""
    import ast, builtins

    global _KNOWN
    if _KNOWN is None:
        _KNOWN = set(dir(builtins)) | _header_names()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"<syntax error>"}
    bound, used = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (used if isinstance(node.ctx, ast.Load) else bound).add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
        elif isinstance(node, ast.MatchAs) and node.name:
            bound.add(node.name)
    return used - bound - _KNOWN


def starter_from(code):
    """LeetCode-style starter code (signatures, empty bodies) derived from a solution: the top-level
    functions and classes except main(), with each class's method signatures -- what a functional
    LiveCodeBench prompt shows, so training prompts can say "use the following starter code" too."""
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    lines = code.splitlines()
    head = lambda n: "\n".join(lines[n.lineno - 1 : n.body[0].lineno - 1]).rstrip()
    parts = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name != "main":
            parts.append(head(node) + "\n    ")
        elif isinstance(node, ast.ClassDef):
            methods = [
                head(m) + "\n        " for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            parts.append(head(node) + "\n" + "\n".join(methods))
    return "\n".join(parts)


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


def _read_and_transform(args):
    """Runs in a worker process (see parallel_parquet_rows): read one parquet file, apply
    `transform` to every row, return the non-None results as a list."""
    import pyarrow.parquet as pq

    file, columns, transform = args
    try:
        pf = pq.ParquetFile(file)
    except Exception as e:
        print(f"parallel_parquet_rows: skipping unreadable (likely still downloading) {file}: {e!r}")
        return []
    return [
        row
        for batch in pf.iter_batches(batch_size=2048, columns=columns)
        for raw in batch.to_pylist()
        for row in [transform(raw)]
        if row is not None
    ]


def parallel_parquet_rows(pattern, transform, columns=None, workers=None):
    """Like parquet_rows, but applies `transform` (a module-level, picklable function: one raw
    pyarrow row -> one canonical row dict, or None to drop it) inside a worker process per file,
    instead of materializing and transforming every row on a single core. Reading rstar-seed's raw
    source this way (one Python loop, no multiprocessing) took over 11 minutes on one of this
    server's 128 cores -- long text fields mean real per-row cost: hashing, a hedge-marker regex,
    normalization. A subset's source is only read in full when its builder_version changes (see
    build()), but that still happens often enough during active data work to be worth parallelizing
    instead of accepted as a fixed cost every time."""
    import os
    from concurrent.futures import ProcessPoolExecutor

    files = sorted(glob.glob(str(ROOT / pattern)))
    workers = min(len(files), workers or os.cpu_count() or 1) or 1
    with ProcessPoolExecutor(workers) as pool:
        for rows in pool.map(_read_and_transform, [(f, columns, transform) for f in files]):
            yield from rows


def _read_jsonl_range(args):
    """Worker for parallel_jsonl_rows: the lines that *start* inside [begin, end)."""
    path, begin, end, transform = args
    rows = []
    with open(path, "rb") as f:
        if begin:
            f.seek(begin - 1)
            f.readline()  # finish the line straddling `begin`; it belongs to the previous range
        while f.tell() < end:
            line = f.readline()
            if not line:
                break
            row = transform(json.loads(line))
            if row is not None:
                rows.append(row)
    return rows


def parallel_jsonl_rows(path, transform, chunks=64, workers=None):
    """parallel_parquet_rows for one large JSON-lines file: split by byte range, transform per line
    in worker processes, yield in file order."""
    import os
    from concurrent.futures import ProcessPoolExecutor

    path = str(ROOT / path)
    size = os.path.getsize(path)
    bounds = [size * i // chunks for i in range(chunks + 1)]
    with ProcessPoolExecutor(min(chunks, workers or os.cpu_count() or 1)) as pool:
        for rows in pool.map(_read_jsonl_range, [(path, bounds[i], bounds[i + 1], transform) for i in range(chunks)]):
            yield from rows


# ---------------------------------------------------------------- source readers
def read_kodcode():
    """KodCode-V1-SFT-4o: synthetic function-style tasks (incl. ones derived from Taco, Codeforces, Apps,
    Code_Contests, Leetcode), GPT-4o reply with explanation + code, verified by unit tests."""
    for r in parquet_rows(
        "data/raw/kodcode-v1-sft-4o/data/train-*.parquet",
        columns=[
            "conversations",
            "subset",
            "question",
            "4o_solution",
            "question_id",
            "style",
            "4o_correctness",
            "gpt_difficulty",
            "gpt_pass_percentage",
        ],
    ):
        reply = next((c["value"] for c in r["conversations"] if c["from"] == "gpt"), "")
        yield {
            "id": f"kodcode/{r['question_id']}",
            "problem_id": r["question_id"],
            "source": r["subset"],
            "problem": r["question"],
            "response": reply,
            "code": last_code(reply) or r["4o_solution"],
            "kind": "function",
            "meta": {
                "style": r["style"],
                "correct": r["4o_correctness"] == "True",
                "gpt_difficulty": r["gpt_difficulty"],
                "pass_rate": r["gpt_pass_percentage"],
            },
        }


def read_opencodereasoning():
    """OpenCodeReasoning split_0: competitive programming (CodeContests, TACO, APPS, Codeforces) with a
    DeepSeek-R1 reply (reasoning + code) and the code alone; several replies per problem, not verified.
    Upstream `id` is shared across a problem's sampled completions, not unique per row (measured up
    to 18 rows sharing one id), so it cannot be the row id; a running counter plus a text digest is."""
    for i, r in enumerate(
        parquet_rows(
            "data/raw/opencodereasoning/split_0/*.parquet",
            columns=["output", "input", "solution", "dataset", "difficulty", "id"],
        )
    ):
        reply = THINK.sub("", r["output"]) if "</think>" in r["output"] else r["output"]
        kind = (
            "stdin"
            if re.search(r"input\(\)|sys\.stdin", r["solution"])
            else "function" if "def " in r["solution"] else "unknown"
        )
        # problem_id must key off the same normalized text as the exact-dedup hash below and
        # mopd.data.pool.validate_holdout, or two whitespace-variant copies of one problem get
        # treated as different problems: one can then land in train and the other in dev, which
        # validate_holdout then (correctly) rejects as a literal prompt collision.
        yield {
            "id": f"ocr/{i}-{text_hash(r['input'] + chr(0) + r['output'])[:12]}",
            "problem_id": text_hash(r["input"])[:16],
            "source": r["dataset"],
            "problem": r["input"],
            "response": r["output"],
            "code": last_code(r["solution"]) or r["solution"],
            "kind": kind,
            "meta": {
                "difficulty": r["difficulty"],
                "trace_chars": len(r["output"]),
                "final_chars": len(reply),
                "upstream_id": r["id"],
            },
        }


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
    twice, which mopd.data.pool.build_plan then (correctly) rejects as a training-pool collision.
    Reads its ~20 parquet files in parallel (parallel_parquet_rows): the per-row work below (two
    hashes, a language regex, a hedge-marker regex over a long response) is pure and stateless, so
    there is nothing sequential about doing it 591k times -- see parallel_parquet_rows's docstring
    for why that mattered here."""
    return parallel_parquet_rows("data/raw/rstar-coder/seed_sft/*.parquet", _rstar_row)


def _rstar_row(r):
    """Per-row transform for read_rstar_seed; module-level so parallel_parquet_rows can pickle it
    to worker processes. Row id no longer needs a running counter for uniqueness (dropped when this
    moved off a single sequential loop): the two source fields hashed together are already enough
    entropy that a collision across 591k rows is not a practical concern."""
    kind = "function" if r["starter_code"].strip() else "stdin"
    return {
        "id": f"rstar/{text_hash(r['question'] + chr(0) + r['response'])[:16]}",
        "problem_id": text_hash(r["question"])[:16],
        "source": "rstar-coder-seed",
        "problem": r["question"],
        "response": r["response"],
        "code": r["code"] or last_code(r["response"]),
        "kind": kind,
        "meta": {
            "verified": bool(r["verified"]),
            "is_passed": bool(r["is_passed"]),
            "clean_language": not bool(NONLATIN.search(r["response"])),
            "hedge_markers": hedge_count(r["response"]),
            "starter": r["starter_code"].strip() if kind == "function" else "",
            "self_contained": not undefined_names(r["code"] or last_code(r["response"])),
        },
    }


OCI_FILE = "data/raw/opencodeinstruct-clean-low/opencodeinstruct-clean-low-753623-20261015/train.jsonl"
TOP_DEF = re.compile(r"^(?:def|class)\s+(\w+)", re.M)


def read_opencodeinstruct():
    """OpenCodeInstruct, cleaned low-reasoning-effort subset (data/raw/opencodeinstruct-clean-low,
    see its download-manifest.json and README.txt): one Python solution per problem, the reply is a
    short <think> trace (median ~535 supervised tokens in its own tokenizer) then the final code.
    Its bundled DeepSeek-style chat template, generic system prompt and token offsets are dropped:
    recipes render problem/response with this project's prompt formats and the Qwen3 tokenizer,
    like every other subset. kind: "function" when the final code defines anything besides main()
    (the problem asks for a named function or class), "stdin" otherwise."""
    return parallel_jsonl_rows(OCI_FILE, _oci_row)


def _oci_row(r):
    problem = next(m["content"] for m in r["messages"] if m["role"] == "user")
    response = next(m["content"] for m in r["messages"] if m["role"] == "assistant")
    final = response.split("</think>", 1)[-1]
    code = last_code(final)
    kind = "function" if any(name != "main" for name in TOP_DEF.findall(code)) else "stdin"
    return {
        "id": f"oci/{r['id'][:20]}",
        "problem_id": text_hash(problem)[:16],
        "source": "opencodeinstruct",
        "problem": problem,
        "response": response,
        "code": code,
        "kind": kind,
        "meta": {
            "revision": r["revision"],
            "upstream_targets": r["targets"],
            "think_chars": len(response) - len(final),
            "has_code": bool(code),
            "starter": starter_from(code) if kind == "function" else "",
            "self_contained": bool(code) and not undefined_names(code),
        },
    }


def read_smol_constraints():
    """smol-constraints instruction-following SFT pool (the full256 instruction expert's data):
    for mixing into code recipes. No code; kind "instruction" (recipes render it with prompt "plain")."""
    for line in (ROOT / "data/sft/smol_constraints/train.jsonl").open():
        r = json.loads(line)
        yield {
            "id": f"smol/{r['id']}",
            "problem_id": text_hash(r["prompt"])[:16],
            "source": "smol-constraints",
            "problem": r["prompt"],
            "response": r["response"],
            "code": "",
            "kind": "instruction",
            "meta": {},
        }


def read_opencoder_clean():
    """The current code pool (OpenCoder stage 2 educational + evol, opener boilerplate removed)."""
    for line in (ROOT / "data/sft/opencoder_stage2_clean/train.jsonl").open():
        r = json.loads(line)
        yield {
            "id": f"opencoder/{r['id']}",
            "problem_id": str(r["id"]),
            "source": r.get("source", "opencoder"),
            "problem": r["prompt"],
            "response": r["response"],
            "code": last_code(r["response"]),
            "kind": "function",
            "meta": {},
        }


SUBSETS = {
    "kodcode-4o": {"reader": read_kodcode, "raw": "data/raw/kodcode-v1-sft-4o"},
    "opencodereasoning": {"reader": read_opencodereasoning, "raw": "data/raw/opencodereasoning"},
    "opencoder-clean": {"reader": read_opencoder_clean, "raw": "data/sft/opencoder_stage2_clean"},
    "rstar-seed": {"reader": read_rstar_seed, "transform": _rstar_row, "raw": "data/raw/rstar-coder"},
    "opencodeinstruct": {
        "reader": read_opencodeinstruct,
        "transform": _oci_row,
        "raw": "data/raw/opencodeinstruct-clean-low",
    },
    "smol-constraints": {"reader": read_smol_constraints, "raw": "data/sft/smol_constraints"},
}


# ---------------------------------------------------------------- build
def code_digest(obj):
    """Bytes identifying a function's or module's *logic*: its AST without formatting, comments or
    docstrings, so reformatting the file does not invalidate every cached subset and recipe."""
    import ast

    tree = ast.parse(textwrap.dedent(inspect.getsource(obj)))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)) and ast.get_docstring(node):
            node.body = node.body[1:] or [ast.Pass()]
    return ast.dump(tree, include_attributes=False).encode()


def builder_version(name):
    """Digest of this subset's reader (and, for readers that push their per-row logic into a
    separate `transform` for parallel_parquet_rows, that transform too -- otherwise a change to it
    would go undetected since the thin reader wrapper's own source never changes), the shared build
    steps, the decontamination code and the raw download manifest: changing any of them yields a new
    version; other subsets are unaffected."""
    h = hashlib.sha256()
    for part in (
        SUBSETS[name]["reader"],
        SUBSETS[name].get("transform"),
        build,
        last_code,
        undefined_names,
        starter_from,
    ):
        if part is not None:
            h.update(code_digest(part))
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
                counts["empty"] += 1
                continue
            key = hashlib.sha1((" ".join(r["problem"].split()) + "\0" + r["response"]).encode()).digest()
            if key in seen:
                counts["duplicate"] += 1
                continue
            seen.add(key)
            pkey = hashlib.sha1(r["problem"].encode()).digest()  # several replies share one problem
            if pkey not in verdict:
                verdict[pkey] = index.worst(r["problem"])
            pid, cov = verdict[pkey]
            if cov >= decontam.THRESHOLD:
                counts["contaminated"] += 1
                dropped.append([r["id"], pid, round(cov, 3)])
                continue
            counts["kept"] += 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(out / "rows.jsonl")
    sha = hashlib.sha256((out / "rows.jsonl").read_bytes()).hexdigest()
    manifest = {
        "name": name,
        "version": version,
        "rows_sha256": sha,
        "counts": counts,
        "decontam": {"threshold": decontam.THRESHOLD, "eval_files": EVAL_FILES, "dropped": dropped[:2000]},
        "raw": SUBSETS[name]["raw"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=1))
    return manifest


def rows(name, root=ROOT / "data/subsets"):
    manifest = json.loads((root / name / "manifest.json").read_text())
    if manifest["version"] != builder_version(name):
        raise ValueError(f"subset {name} is stale: rebuild it (scripts/subsets.py build {name})")
    with (root / name / "rows.jsonl").open() as f:
        for line in f:
            yield json.loads(line)
