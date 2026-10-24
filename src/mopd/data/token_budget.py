"""Per-step batches sized by trained-token budget instead of sample count (SFT only).

Sample-matched SFT runs train on very different token totals because reference
answers differ in length by domain (math ~780 tokens, instruction ~130). A
token-matched run keeps the step count but fills each step's per-domain token quota
(budget x domain weight) by drawing whole samples from the domain's seeded shuffle
until the quota is reached; the last sample may overshoot it. Draws are without
replacement and fail loudly when a pool runs out, like build_plan.

Sources, holdout check and shuffle seeds are exactly those of mixture.plan_domains
(seed + domain index, single-source domains only), so a domain's draw order is the
same as in the sample-matched plan; only how many rows land in each step differs.
MOPD cannot use this: rollout lengths are unknown before generation.
"""

import random

from mopd.data.mixture import _rows
from mopd.data.pool import load_pool, text_hash, validate_holdout


def plan_by_tokens(config, sample_tokens):
    """sample_tokens(row) -> trained tokens for that row (response incl. EOS, after truncation)."""
    seed, steps = config["seed"], config["steps"]
    budget = config["token_budget"]["per_step"]
    per_step = [[] for _ in range(steps)]
    for i, (domain, spec) in enumerate(config["domains"].items()):
        if len(spec["sources"]) != 1:
            raise ValueError(f"domains.{domain}: token budget supports single-source domains only")
        rows = _rows(spec["sources"][0]["path"], domain, spec["sources"][0]["name"])
        if len({text_hash(r["prompt"]) for r in rows}) != len(rows):
            raise ValueError(f"domains.{domain}: duplicate normalized prompts in pool")
        holdout = [
            dict(r, prompt=r.get("prompt", r.get("problem")))
            for path in spec.get("holdout_files", [])
            for r in load_pool(path)
        ]
        if holdout:
            validate_holdout(rows, holdout)
        if any(r.get("split") != "train" for r in rows):
            raise ValueError(f"domains.{domain}: training pool includes non-training records")
        order = list(rows)
        random.Random(seed + i).shuffle(order)
        quota = budget * spec["weight"]
        cursor = 0
        for step in range(steps):
            filled = 0
            while filled < quota:
                if cursor == len(order):
                    raise ValueError(f"domains.{domain}: pool exhausted at step {step + 1}; no implicit repeats")
                row = order[cursor]
                cursor += 1
                filled += sample_tokens(row)
                per_step[step].append(row)
    return per_step
