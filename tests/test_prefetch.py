import threading, time, unittest

from mopd.rollout.base import Prefetched, RolloutStream


class PrefetchTest(unittest.TestCase):
    def test_drains_before_iteration(self):
        done = threading.Event()

        def source():
            yield from range(3)
            done.set()

        p = Prefetched(RolloutStream(source(), lambda: {"n": 3}))
        self.assertTrue(done.wait(5))  # generated without anyone iterating
        self.assertEqual(list(p), [0, 1, 2])
        self.assertEqual(p.stats, {"n": 3})

    def test_failure_reraises(self):
        def source():
            yield 1
            raise RuntimeError("engine died")

        p = Prefetched(RolloutStream(source()))
        with self.assertRaisesRegex(RuntimeError, "engine died"):
            list(p)


if __name__ == "__main__":
    unittest.main()
