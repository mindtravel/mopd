"""Group trajectories into per-domain micro-batches in arrival order (no torch).

A domain's micro-batch is released as soon as it fills, so scoring and backpropagation can
start while other rollouts are still decoding; whatever is left is released when the stream
ends. Each trajectory lands in exactly one micro-batch, and per-domain counts are checked
against the step plan, so the step's sample set and loss weights do not depend on arrival
order -- only micro-batch composition (a floating-point effect) does.
"""

from collections import Counter, defaultdict
from typing import Iterable, Iterator


class DomainBatcher:
    def __init__(self, micro_batch_size: int, expected_counts: dict[str, int]):
        if micro_batch_size < 1:
            raise ValueError("micro_batch_size must be positive")
        self.size = micro_batch_size
        self.expected = dict(expected_counts)

    def consume(self, trajectories: Iterable[dict]) -> Iterator[tuple[str, list[dict]]]:
        pending = defaultdict(list)
        seen = Counter()
        for trajectory in trajectories:
            domain = trajectory["row"]["domain"]
            if domain not in self.expected:
                raise ValueError(f"trajectory for unplanned domain {domain!r}")
            seen[domain] += 1
            pending[domain].append(trajectory)
            if len(pending[domain]) == self.size:
                yield domain, pending.pop(domain)
        for domain in self.expected:  # config order, deterministic
            if pending.get(domain):
                yield domain, pending.pop(domain)
        if dict(seen) != {d: n for d, n in self.expected.items() if n}:
            raise ValueError(f"step received {dict(seen)} trajectories, planned {self.expected}")
