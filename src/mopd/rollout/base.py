"""One interface for consuming rollouts: an iterator of trajectories plus timing/stats.

The trainer never branches on backend or on streaming. A backend that can stream (vLLM
workers) yields each trajectory as soon as it finishes; any other backend is wrapped so its
batch generate() result is yielded after it returns. Within one step every trajectory comes
from the same policy version and parameters only change after the step, so consuming
trajectories early (scoring and backpropagating them while others still decode) leaves the
step's gradient unchanged up to floating-point accumulation order.
"""

import time
from typing import Callable, Iterable, Iterator


class RolloutStream:
    """Iterate trajectories; `seconds` (until the last one arrived) and `stats` are set when exhausted."""

    def __init__(self, source: Iterable[dict], finish: Callable[[], dict] = dict):
        self._source = source
        self._finish = finish
        self._started = time.perf_counter()
        self.seconds = None
        self.stats = None
        self.count = 0

    def __iter__(self) -> Iterator[dict]:
        for trajectory in self._source:
            self.count += 1
            yield trajectory
        self.seconds = time.perf_counter() - self._started
        self.stats = self._finish()


def open_stream(backend, student, rows, version, *, streaming: bool) -> RolloutStream:
    """Streaming backends expose stream(student, rows, version) -> RolloutStream."""
    if streaming and hasattr(backend, "stream"):
        return backend.stream(student, rows, version)
    trajectories, seconds, stats = backend.generate(student, rows, version)
    stream = RolloutStream(trajectories, lambda: stats)
    stream._started -= seconds  # report the backend's own generation time
    return stream
