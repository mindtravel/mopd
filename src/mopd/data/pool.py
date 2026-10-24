"""Validated training pools and reproducible, no-replacement batch plans."""

import hashlib
import json
import random
from pathlib import Path


def text_hash(text):
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def load_pool(path):
    # JSONL records end at \n only; str.splitlines() would also split on U+2028 inside strings.
    rows = [json.loads(s) for s in Path(path).read_text(encoding="utf-8").split("\n") if s.strip()]
    return rows


def build_plan(rows, *, seed, steps, batch_size):
    if steps < 1 or batch_size < 1:
        raise ValueError("Positive steps and batch size required")
    ids, texts = set(), set()
    for row in rows:
        if row.get("split") != "train":
            raise ValueError("Training pool includes non-training records")
        if row.get("domain") not in ("math", "instruction", "code") or not row.get("prompt", "").strip():
            raise ValueError("Missing prompt or unsupported domain")
        key = (row["domain"], row["id"])
        digest = text_hash(row["prompt"])
        if key in ids or digest in texts:
            raise ValueError("Duplicate ID or normalized prompt")
        ids.add(key)
        texts.add(digest)
    if steps * batch_size > len(rows):
        raise ValueError("Pool exhausted: reduce steps or add data; no implicit repeats")
    ordered = list(rows)
    random.Random(seed).shuffle(ordered)
    return [ordered[i * batch_size : (i + 1) * batch_size] for i in range(steps)]


def validate_holdout(train, holdout):
    overlap = {text_hash(r["prompt"]) for r in train} & {text_hash(r["prompt"]) for r in holdout}
    if overlap:
        raise ValueError("Training/holdout prompt overlap")
