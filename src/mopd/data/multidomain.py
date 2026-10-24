"""Fixed per-domain quotas and domain-mean loss weighting."""

import math
from mopd.data.pool import build_plan


def balanced_plan(pools, counts, *, seed, steps):
    if set(pools) != set(counts) or any(n < 1 for n in counts.values()):
        raise ValueError("Positive quota required for every pool")
    plans = {}
    for i, (domain, rows) in enumerate(pools.items()):
        if any(r["domain"] != domain for r in rows):
            raise ValueError("Misrouted domain")
        plans[domain] = build_plan(rows, seed=seed + i, steps=steps, batch_size=counts[domain])
    return [[r for domain in pools for r in plans[domain][step]] for step in range(steps)]


def domain_scale(domain, weights, counts):
    if (
        set(weights) != set(counts)
        or not math.isclose(sum(weights.values()), 1.0)
        or any(w <= 0 for w in weights.values())
        or any(n <= 0 for n in counts.values())
    ):
        raise ValueError("Positive counts and normalized positive domain weights required")
    return weights[domain] / counts[domain]
