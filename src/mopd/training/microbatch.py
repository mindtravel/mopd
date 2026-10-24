"""Group a rank's encoded samples into micro-batches by padded-token budget.

Samples are sorted longest first, so a micro-batch's width is its first member's length and its
padded size is width * count. A budget in tokens (instead of a fixed count) makes long samples run
alone and short ones run many at a time: the peak memory is bounded by the budget, not by the
longest sample times the count (the OOM mode of 2026-10-24), and short-sample recipes stop
under-filling the GPU. Pure Python so the policy is unit-testable without a GPU.
"""


def by_budget(encoded, length, max_count, max_tokens=None):
    """encoded: items; length(item) -> sequence length. Yields lists of items, longest first.
    max_count caps a micro-batch's size; max_tokens (None = unlimited) caps width * count."""
    items = sorted(encoded, key=length, reverse=True)
    batch = []
    for item in items:
        width = length(batch[0]) if batch else length(item)
        if batch and (len(batch) >= max_count or (max_tokens and width * (len(batch) + 1) > max_tokens)):
            yield batch
            batch = []
        batch.append(item)
    if batch:
        yield batch
