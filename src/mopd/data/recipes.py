"""Recipes: a small JSON that turns subsets into one SFT training pool.

  configs/recipes/<name>.json
  {"name": "...", "description": "...",
   "parts": [{"subset": "kodcode-4o", "where": {"source": ["Taco", "Codeforces"], "meta.correct": [true]},
              "weight": 1.0}],
   "per_problem": 1,                     # at most this many replies per problem
   "prefer_fewest": "meta.hedge_markers",# optional; of a problem's shuffled candidates, keep the
                                          # `per_problem` with the lowest value of this field first
                                          # (e.g. subsets.hedge_count), not a random pick
   "prompt": "lcb" | "lcb_think" | "plain",      # lcb: LiveCodeBench v1 wording; lcb_think: lcb-dev-v2-think
   "response": "full" | "final" | "code" | "think" | "nothink",  # full reply / reply without <think> /
                                          # code only / reasoning + </think> + answer / </think> + answer
                                          # (think, nothink: for plain_qa_think_v1 students)
   "max_total_tokens": 20480,            # prompt + response + EOS above this is skipped (student
                                          # tokenizer); benchmark_seq_len.py found real data tops
                                          # out at ~19.7k, and the cost is in micro-batching two
                                          # long sequences together, not in the cap itself
   "rows": 8192, "dev_rows": 64, "seed": 20261022,
   "micro_batch_size": 2}                # per-recipe override for the sweep's SFT config; use 1
                                          # for recipes with long (e.g. full reasoning-chain) replies

A part may override "prompt" and "response" for its own rows.
where: field -> list of allowed values, or {"min": x, "max": y}; "meta.k" reads row["meta"]["k"].
Each part gets rows * weight / sum(weights) rows, drawn without replacement after a seeded shuffle.
Output data/recipes/<name>/{train.jsonl, dev.jsonl, manifest.json} in the SFT pool format
(id, domain, source, prompt, response, split). The version covers the recipe file, the subset
versions and this module; an unchanged recipe is not rebuilt.
"""
import hashlib, inspect, json, random
from mopd.data.pool import text_hash  # same normalization used to defend the dev/train split below
from pathlib import Path

from mopd.data import subsets
from mopd.evaluation import livecodebench as lcb

ROOT = subsets.ROOT
FUNCTION_FORMAT = "Reply with the complete Python function in a single ```python code block."


def field(row, key):
    if key.startswith("meta."):
        return row["meta"].get(key[5:])
    return row[key]


def matches(row, where):
    for key, cond in (where or {}).items():
        v = field(row, key)
        if isinstance(cond, dict):
            if v is None or ("min" in cond and v < cond["min"]) or ("max" in cond and v > cond["max"]):
                return False
        elif v not in cond:
            return False
    return True


def format_prompt(row, style):
    if style == "plain":
        return row["problem"]
    if style == "lcb":
        if row["kind"] == "stdin":
            return lcb.prompt(row["problem"], "stdin")
        if row["meta"].get("starter"):  # exactly how a functional LiveCodeBench problem is asked
            return lcb.prompt(row["problem"], "functional", row["meta"]["starter"])
        return lcb.INSTRUCTION.format(question=row["problem"]) + FUNCTION_FORMAT
    if style == "lcb_think":  # lcb-dev-v2-think wording; pair with student_prompt_format plain_qa_think_v1
        if row["kind"] == "function" and row["meta"].get("starter"):
            return lcb.prompt_think(row["problem"], "functional", row["meta"]["starter"])
        return lcb.prompt_think(row["problem"], "stdin")
    raise ValueError(f"unknown prompt style {style!r}")


def format_response(row, style):
    if style == "full":
        return row["response"]
    if style == "final":
        return subsets.THINK.sub("", row["response"]) if "</think>" in row["response"] else row["response"]
    if style == "code":
        return f"```python\n{row['code'].strip()}\n```" if row["code"].strip() else ""
    if style == "nothink":  # the think protocol's no-reasoning control: close the block, then answer
        code = row["code"].strip()
        answer = f"```python\n{code}\n```" if code else row["response"].replace("<think>", "").split("</think>", 1)[-1].strip()
        return "</think>\n\n" + answer
    if style == "think":
        # Continues the prompt's open <think>: reasoning, then </think>, then the answer -- the code
        # block for code rows, the reply itself for rows without code (instruction data mixed in).
        # Reasoning = the part before </think> if the reply has one, else the reply minus its final
        # code block; empty reasoning closes the block at once (a "no-think" control).
        text = row["response"].replace("<think>", "")
        code = row["code"].strip()
        if "</think>" in text:
            reasoning, answer = text.split("</think>", 1)
        elif code:
            reasoning, answer = lcb_strip_final_code(text), ""
        else:
            reasoning, answer = "", text
        answer = f"```python\n{code}\n```" if code else answer.strip()
        reasoning = reasoning.strip()
        return (reasoning + "\n</think>\n\n" if reasoning else "</think>\n\n") + answer
    raise ValueError(f"unknown response style {style!r}")


def lcb_strip_final_code(text):
    """Drop the last fenced code block (and anything after it) from a reply."""
    blocks = list(subsets.FENCE.finditer(text))
    return text[:blocks[-1].start()] if blocks else text


def version(recipe):
    h = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode())
    h.update(inspect.getsource(inspect.getmodule(version)).encode())
    for part in recipe["parts"]:
        h.update(subsets.builder_version(part["subset"]).encode())
    return h.hexdigest()[:12]


def build(recipe_path, out_root=ROOT / "data/recipes"):
    recipe = json.loads(Path(recipe_path).read_text())
    out = out_root / recipe["name"]
    ver = version(recipe)
    if (out / "manifest.json").exists() and json.loads((out / "manifest.json").read_text())["version"] == ver:
        return json.loads((out / "manifest.json").read_text())
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / "models/Qwen3-4B-Base"))
    rng = random.Random(recipe["seed"])
    total_w = sum(p.get("weight", 1.0) for p in recipe["parts"])
    want_total = recipe["rows"] + recipe.get("dev_rows", 64)
    picked, report = [], []
    for i, part in enumerate(recipe["parts"]):
        want = round(want_total * part.get("weight", 1.0) / total_w)
        by_problem = {}
        for r in subsets.rows(part["subset"]):
            if matches(r, part.get("where")):
                by_problem.setdefault(r["problem_id"], []).append(r)
        prefer = part.get("prefer_fewest", recipe.get("prefer_fewest"))
        candidates = []
        for pid in sorted(by_problem):
            group = by_problem[pid]
            rng.shuffle(group)  # random order; stable sort below keeps this as the tiebreak
            if prefer:
                group.sort(key=lambda r: field(r, prefer))
            candidates += group[:recipe.get("per_problem", 1)]
        rng.shuffle(candidates)
        taken, too_long, empty = [], 0, 0
        for r in candidates:
            if len(taken) == want:
                break
            response = format_response(r, part.get("response", recipe["response"]))
            if not response.strip():
                empty += 1; continue
            prompt = format_prompt(r, part.get("prompt", recipe["prompt"]))
            total = len(tok(prompt, add_special_tokens=False).input_ids) + len(tok(response, add_special_tokens=False).input_ids) + 1
            if total > recipe["max_total_tokens"]:
                too_long += 1; continue
            taken.append({"id": f"{recipe['name']}/{r['id']}", "domain": "code", "source": f"{part['subset']}:{r['source']}",
                          "prompt": prompt, "response": response, "split": "train",
                          "problem_id": f"{part['subset']}/{r['problem_id']}"})
        if len(taken) < want:
            raise ValueError(f"part {i} ({part['subset']}): only {len(taken)} of {want} rows after filters")
        picked += taken
        report.append({"subset": part["subset"], "where": part.get("where"), "problems": len(by_problem),
                       "candidates": len(candidates), "taken": len(taken), "skipped_too_long": too_long, "skipped_empty": empty,
                       "subset_version": subsets.builder_version(part["subset"])})
    rng.shuffle(picked)
    # dev rows: whole problems held out, so no dev problem appears in train
    # Hold out by problem_id (upstream's own grouping) AND by normalized prompt text, in case two
    # rows that render to the same prompt got different problem_ids upstream -- validate_holdout
    # would reject that collision at training time anyway, but this keeps it out of dev too.
    dev_problems, dev_prompts, dev = set(), set(), []
    for r in picked:
        if len(dev) >= recipe.get("dev_rows", 64):
            break
        dev_problems.add(r["problem_id"]); dev_prompts.add(text_hash(r["prompt"])); dev.append(r)
    train = [r for r in picked if r["problem_id"] not in dev_problems and text_hash(r["prompt"]) not in dev_prompts][:recipe["rows"]]
    if len(train) < recipe["rows"]:
        raise ValueError(f"only {len(train)} train rows after holding out dev problems")
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("dev", dev)):
        (out / f"{name}.jsonl").write_text("".join(json.dumps({**r, "split": "train" if name == "train" else "dev"},
                                                               ensure_ascii=False) + "\n" for r in rows))
    manifest = {"name": recipe["name"], "version": ver, "recipe": recipe, "parts": report,
                "train_rows": len(train), "dev_rows": len(dev),
                "train_sha256": hashlib.sha256((out / "train.jsonl").read_bytes()).hexdigest(),
                "dev_sha256": hashlib.sha256((out / "dev.jsonl").read_bytes()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    return manifest
