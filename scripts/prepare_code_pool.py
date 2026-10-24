"""Build a code SFT pool from OpenCoder parquet subsets, decontaminated against HumanEval/MBPP.

Defaults reproduce the original stage-1 pool (data/sft/opencoder: filtered_infinity_instruct,
ids "opencoder_filtered_infinity-<i>"). --set stage2 builds data/sft/opencoder_stage2 from the
code-focused stage-2 subsets educational_instruct + evol_instruct.

Decontamination, as for the instruction pools vs IFEval: normalized exact-match plus 8-gram
content overlap with any HumanEval / MBPP (all sanitized splits) prompt, ignoring n-grams that
recur in >= 5 training rows (boilerplate, not real overlap). Stage-2 educational_instruct has
the same "write a function" shape as MBPP, so rows whose entry point equals any benchmark
function name are dropped as well (conservative: also drops unrelated same-name tasks).
Single-turn pairs only; prompt/response capped at 1024 tokens; an independent 256-row dev split.
"""

import argparse, ast, collections, hashlib, json, random, re
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from mopd.data.pool import text_hash, validate_holdout
from mopd.protocol import student_prompt

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / "data/code/upstream"
SETS = {
    "stage1": {
        "out": "data/sft/opencoder",
        "license": "apache-2.0",
        "revision": "1bcab575f5e2d1c1fd6652720418524c27b3d58b",
        "sources": [
            (
                "OpenCoder-LLM/opc-sft-stage1:filtered_infinity_instruct",
                "opencoder_filtered_infinity",
                UP / "opc_sft_stage1/filtered_infinity_instruct-00000-of-00004.parquet",
            )
        ],
    },
    "stage2": {
        "out": "data/sft/opencoder_stage2",
        "license": "mit",
        "revision": "7d28f40d579edd7c24402d17d0c7639f991e6f8d",
        "sources": [
            (
                "OpenCoder-LLM/opc-sft-stage2:educational_instruct",
                "opc2_educational",
                UP / "opc_sft_stage2/educational_instruct-train-00000-of-00001.parquet",
            ),
            (
                "OpenCoder-LLM/opc-sft-stage2:evol_instruct",
                "opc2_evol",
                UP / "opc_sft_stage2/evol_instruct-train-00000-of-00001.parquet",
            ),
        ],
    },
}
# educational_instruct alone: function-level Python with tests, the subset closest to HumanEval / MBPP.
SETS["stage2edu"] = {**SETS["stage2"], "out": "data/sft/opencoder_stage2_edu", "sources": SETS["stage2"]["sources"][:1]}
# stage2 with the leading boilerplate sentence removed from responses (see wash_opener).
SETS["stage2clean"] = {**SETS["stage2"], "out": "data/sft/opencoder_stage2_clean", "wash": True}
# educational_instruct prefixes every response with this fixed template line; evol_instruct (GPT-4 answers via
# evol-codealpaca-v1 -> Magicoder-Evol-Instruct-110K) often opens with "Certainly! ..." / "Here is a solution:".
# Under MOPD the opener spread to instruction prompts (IFEval responses opening with it: 1.5% -> 29.6%).
_EDU_OPENER = re.compile(r"^\s*Here is the code to solve this problem:\s*\n", re.I)
_LEAD_OPENER = re.compile(
    r"""^\s*(?:
    (?:certainly|sure|of\ course|absolutely|great\ question)[!.,]?[^\n]{0,160}?(?:[:.!]\s*\n|[.!]\s+(?=[A-Z`]))
  | here(?:'s|\ is|\ are)\b[^\n]{0,160}?:\s*\n
)""",
    re.I | re.X,
)


def wash_opener(text):
    """Drop ONE leading boilerplate sentence when substantive content follows; content itself untouched.
    Returns (new_text, removed_sentence_or_None)."""
    m = _EDU_OPENER.match(text) or _LEAD_OPENER.match(text)
    if not m:
        return text, None
    rest = text[m.end() :].lstrip("\n")
    if len(rest.strip()) < 20:
        return text, None
    return rest, text[: m.end()].strip()


MAX_PROMPT_TOKENS = 1024
MAX_RESPONSE_TOKENS = 1024
DEV_SIZE = 256
SEED = 20261021


def norm(s):
    return re.sub(r"\W+", " ", s.lower()).strip()


def grams(s, n=8):
    w = norm(s).split()
    return {" ".join(w[i : i + n]) for i in range(len(w) - n + 1)}


def load_benchmarks():
    prompts, names = [], set()
    he = pq.read_table(UP / "humaneval/test-00000-of-00001.parquet", columns=["prompt", "entry_point"]).to_pylist()
    prompts += [r["prompt"].strip("'\"") for r in he]
    names |= {r["entry_point"].strip("'\"") for r in he}
    for split in ("prompt", "test", "train", "validation"):
        mb = pq.read_table(
            UP / f"mbpp/sanitized-{split}-00000-of-00001.parquet", columns=["prompt", "code"]
        ).to_pylist()
        prompts += [r["prompt"] for r in mb]
        names |= {m.group(1) for r in mb for m in [re.search(r"^def\s+(\w+)\s*\(", r["code"], re.M)] if m}
    return prompts, {n.lower() for n in names}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", choices=sorted(SETS), default="stage1")
    a = p.parse_args()
    cfg = SETS[a.set]
    out_root = ROOT / cfg["out"]
    if (out_root / "train.jsonl").exists():
        raise SystemExit(f"{out_root} exists; pools are immutable once written")
    tok = AutoTokenizer.from_pretrained(str(ROOT / "models/Qwen3-4B-Base"), local_files_only=True)
    raw = [
        (source, prefix, i, r)
        for source, prefix, path in cfg["sources"]
        for i, r in enumerate(pq.read_table(path).to_pylist())
    ]
    benchmark_prompts, benchmark_names = load_benchmarks()
    df = collections.Counter(g for *_, r in raw for g in grams(r["instruction"] or ""))
    template = {g for g, c in df.items() if c >= 5}
    benchmark_grams = set().union(*(grams(x) for x in benchmark_prompts)) - template
    seen = set()
    kept = []
    excluded = collections.Counter()
    by_source = collections.Counter()
    washed = collections.Counter()
    removed_lines = collections.Counter()
    for source, prefix, i, r in raw:
        prompt = (r.get("instruction") or "").strip()
        response = (r.get("output") or "").strip()
        if not prompt or not response:
            excluded["empty_field"] += 1
            continue
        if cfg.get("wash"):
            response, cut = wash_opener(response)
            if cut:
                washed[source] += 1
                removed_lines[cut[:80]] += 1
        h = text_hash(prompt)
        if h in seen:
            excluded["duplicate_prompt"] += 1
            continue
        if grams(prompt) & benchmark_grams:
            excluded["benchmark_content_8gram"] += 1
            continue
        entry = (r.get("entry_point") or "").strip()
        if entry and entry.lower() in benchmark_names:
            excluded["benchmark_entry_point_name"] += 1
            continue
        meta = {k: r[k] for k in ("tag", "entry_point") if r.get(k)}
        row = {
            "id": f"{prefix}-{i}",
            "domain": "code",
            "source": source,
            "prompt": prompt,
            "response": response,
            "split": "train",
            "metadata": meta,
        }
        if len(tok(student_prompt(tok, row, "plain_qa_v1")).input_ids) > MAX_PROMPT_TOKENS:
            excluded["prompt_over_max_tokens"] += 1
            continue
        if len(tok(response, add_special_tokens=False).input_ids) > MAX_RESPONSE_TOKENS:
            excluded["response_over_max_tokens"] += 1
            continue
        seen.add(h)
        kept.append(row)
        by_source[source] += 1
    random.Random(SEED).shuffle(kept)
    dev = [dict(r, split="dev") for r in kept[:DEV_SIZE]]
    train = kept[DEV_SIZE:]
    validate_holdout(train, dev)
    out_root.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("dev", dev)):
        (out_root / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    report = {
        "set": a.set,
        "sources": [s for s, *_ in cfg["sources"]],
        "revision": cfg["revision"],
        "license": cfg["license"],
        "raw": len(raw),
        "kept_by_source": dict(by_source),
        "train": len(train),
        "dev": len(dev),
        "excluded": dict(excluded),
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "max_response_tokens": MAX_RESPONSE_TOKENS,
        "decontamination": "HumanEval (164) + MBPP sanitized all splits prompts: normalized exact-match dedup, 8-gram content "
        "overlap excluding n-grams shared by >=5 training rows, and entry-point name equal to any benchmark "
        "function name; semantic paraphrase and model pretraining contamination not ruled out",
        **(
            {
                "washed_leading_opener": {
                    "rule": "scripts/prepare_code_pool.py wash_opener",
                    "rows_by_source": dict(washed),
                    "top_removed": removed_lines.most_common(15),
                }
            }
            if cfg.get("wash")
            else {}
        ),
        "train_sha256": hashlib.sha256((out_root / "train.jsonl").read_bytes()).hexdigest(),
    }
    (out_root / "preparation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
