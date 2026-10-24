"""Regression test: an idle worker must claim queued work before a slow peer finishes."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "scripts")]
import threading, unittest
from mopd.rollout.dispatch import length_bucket_dispatch


class DispatchTests(unittest.TestCase):
    def test_fast_worker_drains_queue(self):
        slow_started = threading.Event()
        release = threading.Event()
        seen = []
        active = set()
        lock = threading.Lock()

        def run(worker, batch):
            with lock:
                self.assertNotIn(worker, active)
                active.add(worker)
            try:
                if batch == [0]:
                    slow_started.set()
                    self.assertTrue(release.wait(3), "idle worker did not drain queue")
                else:
                    self.assertTrue(slow_started.wait(3))
                    if batch == [5]:
                        release.set()
                with lock:
                    seen.extend(batch)
                return batch
            finally:
                with lock:
                    active.remove(worker)

        replies, stats = length_bucket_dispatch(list(range(6)), workers=2, batch_size=1, length=lambda x: x, run=run)
        self.assertEqual(sorted(seen), list(range(6)))
        self.assertEqual(sorted(x for b in replies for x in b), list(range(6)))
        self.assertEqual(stats.batches, 6)

    def test_exception_is_propagated(self):
        def fail(worker, batch):
            raise RuntimeError("worker failed")

        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            length_bucket_dispatch([1], workers=1, batch_size=1, length=lambda x: x, run=fail)


if __name__ == "__main__":
    unittest.main()
