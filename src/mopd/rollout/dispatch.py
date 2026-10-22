"""Length-aware dynamic dispatch for rollout workers.
Workers are independent, single-request-at-a-time engines. This scheduler keeps a
small queue of batches and assigns the next batch to whichever worker finishes.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar
T = TypeVar("T")
R = TypeVar("R")
@dataclass(frozen=True)
class DispatchStats:
    batches: int
    workers: int
    max_batch: int
    min_batch: int
def length_sorted_batches(rows: Sequence[T], *, batch_size: int, length: Callable[[T], int]) -> list[list[tuple[int, T]]]:
    """Rows sorted by length (ties by position) cut into batches of (position, row)."""
    ordered = sorted(enumerate(rows), key=lambda x: (length(x[1]), x[0]))
    return [ordered[i:i + batch_size] for i in range(0, len(ordered), batch_size)]


def length_bucket_dispatch(rows: Sequence[T], *, workers: int, batch_size: int,
                           length: Callable[[T], int],
                           run: Callable[[int, list[T]], R]) -> tuple[list[R], DispatchStats]:
    if workers < 1 or batch_size < 1:
        raise ValueError("workers and batch_size must be positive")
    if not rows:
        return [], DispatchStats(0, workers, 0, 0)
    batches = length_sorted_batches(rows, batch_size=batch_size, length=length)
    results: list[tuple[int, R]] = []
    next_batch = 0
    with ThreadPoolExecutor(max_workers=min(workers, len(batches))) as pool:
        active = {}
        for worker_id in range(min(workers, len(batches))):
            batch = batches[next_batch]; next_batch += 1
            fut = pool.submit(run, worker_id, [x[1] for x in batch])
            active[fut] = (worker_id, batch)
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for fut in done:
                worker_id, batch = active.pop(fut)
                results.append((batch[0][0], fut.result()))
                if next_batch < len(batches):
                    new_batch = batches[next_batch]; next_batch += 1
                    nfut = pool.submit(run, worker_id, [x[1] for x in new_batch])
                    active[nfut] = (worker_id, new_batch)
    results.sort(key=lambda x: x[0])
    sizes = [len(b) for b in batches]
    return [x[1] for x in results], DispatchStats(
        batches=len(batches), workers=min(workers, len(batches)),
        max_batch=max(sizes), min_batch=min(sizes))

