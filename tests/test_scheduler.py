"""mopd.runtime.scheduler: GPUPool fair shares that do not depend on arrival order, and the job DAG.

Two real bugs this guards against:
  - eval callers unpack `gpu, = pool.want(min_gpus=1)` expecting exactly one GPU; without a
    ceiling, a lone caller on an otherwise-idle pool got handed every free GPU and the unpack
    crashed -- after a from-scratch 256-step, 7-GPU training run had already finished.
  - sizing the share from "how many callers are blocked inside want() right now" is racy: whoever
    arrives first sees nobody else waiting yet and grabs everything, silently serializing jobs
    that were meant to run in parallel. Seeding the denominator with the known job count up front,
    and only shrinking it via done() when a job verifiably finishes, removes the race.
  - the denominator also counted jobs that already held a grant: with 2 jobs on 8 GPUs, the first
    took 4 and the second was offered 4 // 2 = 2 of the remaining 4 -- too few for a full
    fine-tuning run, which then failed at start-up.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "scripts")]
import threading
import time
import unittest

import sys
from pathlib import Path
from mopd.runtime.scheduler import GPUPool, Scheduler


class GPUPoolTest(unittest.TestCase):
    def test_lone_job_takes_every_idle_gpu(self):
        pool = GPUPool(range(7))
        self.assertEqual(sorted(pool.want("a")), list(range(7)))

    def test_max_gpus_caps_a_lone_job(self):
        pool = GPUPool(range(7))
        self.assertEqual(len(pool.want("a", min_gpus=1, max_gpus=1)), 1)  # `gpu, = ...` used to crash on 7

    def test_jobs_enqueued_together_split_evenly_whatever_their_thread_order(self):
        pool = GPUPool(range(7))
        got = {}
        for i in range(7):
            pool.enqueue(f"j{i}")
        threads = [threading.Thread(target=lambda i=i: got.__setitem__(i, pool.want(f"j{i}"))) for i in range(7)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(sorted(len(g) for g in got.values()), [1] * 7)
        self.assertEqual(sorted(g for v in got.values() for g in v), list(range(7)))

    def test_blocked_jobs_do_not_dilute_the_share(self):
        pool = GPUPool(range(8))
        pool.enqueue("ready")  # a dependent job that is not ready is never enqueued
        self.assertEqual(len(pool.want("ready")), 8)

    def test_waiting_job_gets_gpus_back_when_one_finishes(self):
        pool = GPUPool(range(4))
        pool.enqueue("a")
        pool.enqueue("b")
        a = pool.want("a")
        b = pool.want("b")
        self.assertEqual((len(a), len(b)), (2, 2))
        done = []
        t = threading.Thread(target=lambda: done.append(pool.want("c", min_gpus=3)))
        t.start()
        time.sleep(0.05)
        self.assertEqual(done, [])  # only 0 free: c waits
        pool.give_back(a)
        pool.give_back(b)
        t.join(2)
        self.assertEqual(len(done[0]), 4)


class UsableGpusTest(unittest.TestCase):
    def test_busy_gpu_is_usable_only_while_one_of_our_leases_holds_it(self):
        import tempfile
        from unittest import mock
        from mopd.runtime import scheduler as S

        d = tempfile.mkdtemp()
        with mock.patch("mopd.runtime.gpus.gpu_memory_used", return_value={0: 0, 1: 20000, 2: 20000}):
            ours = S.GpuLease(d)
            self.assertTrue(ours.try_take([1]))  # GPU 1 busy under our lease, GPU 2 busy and unleased
            self.assertEqual(S.usable_gpus([0, 1, 2], lease_dir=d), [0, 1])
            ours.release([1])


class SchedulerTest(unittest.TestCase):
    def test_dag_runs_deps_first_blocks_on_failure_and_skips_done(self):
        import tempfile

        order, log = [], tempfile.mkdtemp()
        sched = Scheduler([0, 1, 2, 3], log, min_free_gib=0, name="t", poll_seconds=0.05, lease_dir=tempfile.mkdtemp())
        sched.add({"id": "a", "gpus": {"min": 1}, "run": lambda g: order.append(("a", len(g)))})
        sched.add({"id": "b", "deps": ["a"], "gpus": 0, "run": lambda g: order.append(("b", len(g)))})
        sched.add(
            {"id": "c", "deps": ["a"], "gpus": {"min": 1}, "run": lambda g: (_ for _ in ()).throw(RuntimeError("boom"))}
        )
        sched.add({"id": "d", "deps": ["c"], "gpus": {"min": 1}, "run": lambda g: order.append(("d", len(g)))})
        sched.add({"id": "e", "gpus": {"min": 1}, "done": lambda: True, "run": lambda g: order.append(("e", len(g)))})
        state = sched.run()
        self.assertEqual([k for k, _ in order][:1], ["a"])
        self.assertEqual(dict(order)["a"], 4)  # a lone ready job takes every idle GPU
        self.assertEqual(dict(order)["b"], 0)
        self.assertEqual(state["c"]["state"], "failed")
        self.assertEqual(state["d"]["state"], "blocked")
        self.assertTrue(state["e"].get("skipped"))
        self.assertNotIn("d", dict(order))


class AfterTest(unittest.TestCase):
    def test_after_runs_even_when_its_predecessor_failed_deps_do_not(self):
        import tempfile

        ran = []
        sched = Scheduler(
            [0, 1], tempfile.mkdtemp(), min_free_gib=0, name="t", poll_seconds=0.05, lease_dir=tempfile.mkdtemp()
        )
        sched.add({"id": "a", "gpus": 0, "run": lambda g: (_ for _ in ()).throw(RuntimeError("boom"))})
        sched.add({"id": "b", "after": ["a"], "gpus": 0, "run": lambda g: ran.append("b")})
        sched.add({"id": "c", "deps": ["a"], "gpus": 0, "run": lambda g: ran.append("c")})
        state = sched.run()
        self.assertEqual(ran, ["b"])
        self.assertEqual(state["c"]["state"], "blocked")


class RoleTest(unittest.TestCase):
    def test_eval_role_stays_within_eval_gpus(self):
        from mopd.runtime.scheduler import GPUPool

        pool = GPUPool([0, 1, 2, 3, 4, 5])
        self.assertEqual(pool.want("e", 1, None, within=(4, 5)), [4, 5])  # all idle GPUs, but only the role's
        self.assertEqual(pool.want_exactly("t", [0, 1, 2, 3]), [0, 1, 2, 3])  # training still gets its four


if __name__ == "__main__":
    unittest.main()
