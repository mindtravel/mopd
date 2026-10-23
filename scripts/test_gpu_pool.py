"""run_sweep.GPUPool: dynamic fair-share allocation that does not depend on arrival order.

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
import threading
import time
import unittest

from run_sweep import GPUPool


class GPUPoolTest(unittest.TestCase):
    def test_lone_job_takes_every_idle_gpu(self):
        pool = GPUPool(range(7), jobs=1)
        self.assertEqual(sorted(pool.want()), list(range(7)))

    def test_job_already_holding_gpus_does_not_shrink_the_next_share(self):
        pool = GPUPool(range(8), jobs=2)
        self.assertEqual(len(pool.want()), 4)
        self.assertEqual(len(pool.want()), 4)  # the bug: 2, leaving 2 GPUs idle

    def test_max_gpus_caps_a_lone_job(self):
        pool = GPUPool(range(7), jobs=1)
        got = pool.want(min_gpus=1, max_gpus=1)
        self.assertEqual(len(got), 1)  # the bug: this used to be 7, and `gpu, = ...` crashed

    def test_jobs_arriving_in_quick_succession_still_split_evenly(self):
        # Seven jobs, no barrier -- threads start and call want() at whatever pace the OS
        # scheduler gives them, closely but not simultaneously. A "currently waiting" denominator
        # would let an early arrival see nobody else waiting yet and grab everything; the known
        # job count seeded up front must not.
        pool = GPUPool(range(7), jobs=7)
        results = {}

        def worker(i):
            results[i] = pool.want()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(7)]
        for t in threads: t.start()
        for t in threads: t.join(timeout=2)
        self.assertTrue(all(not t.is_alive() for t in threads))  # no deadlock
        sizes = sorted(len(g) for g in results.values())
        self.assertEqual(sizes, [1] * 7)  # matches the old one-GPU-per-variant behavior
        self.assertEqual(sorted(g for gs in results.values() for g in gs), list(range(7)))

    def test_done_shrinks_the_pool_for_whoever_is_left(self):
        pool = GPUPool(range(7), jobs=7)
        gpus = [pool.want() for _ in range(7)]  # one job finishes its only want(), frees nothing
        self.assertEqual(sorted(len(g) for g in gpus), [1] * 7)
        for g in gpus[1:]:
            pool.give_back(g)
        for _ in range(6):
            pool.done()  # six of seven jobs are entirely finished
        last = pool.want()  # the seventh asks again (e.g. its eval phase)
        self.assertEqual(len(last), 6)  # everyone else's GPUs, since only this job remains

    def test_finishing_job_frees_gpus_for_the_one_still_waiting(self):
        pool = GPUPool(range(4), jobs=1)
        a = pool.want()  # only job registered so far: takes all 4
        self.assertEqual(len(a), 4)
        released = threading.Event()

        def waiter():
            b = pool.want(min_gpus=1, max_gpus=1)
            self.assertTrue(released.is_set())  # must not return before a's GPUs are given back
            pool.give_back(b)

        t = threading.Thread(target=waiter); t.start()
        time.sleep(0.05)  # let the waiter block inside want()
        released.set()
        pool.give_back(a)
        t.join(timeout=2)
        self.assertFalse(t.is_alive())


if __name__ == "__main__":
    unittest.main()
