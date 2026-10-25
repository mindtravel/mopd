"""mopd.pool allocation decisions without GPUs."""

import sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mopd.pool.allocate import decide

TRAIN = {"id": "train", "gpus": {"count": 4}, "priority": 50}
EVAL = lambda i, p=10: {"id": f"eval{i}", "gpus": {"min": 1, "max": 4}, "priority": p, "preemptible": True}


class AllocateTest(unittest.TestCase):
    def test_gang_waits_and_backfill_fills_the_rest(self):
        # 2 GPUs busy with a non-preemptible job: training waits; an eval backfills the 2 free GPUs
        starts, preempts = decide([1, 3], {"other": {"gpus": [0, 2], "priority": 50}}, [TRAIN, EVAL(1)])
        self.assertEqual(starts, {"eval1": [1, 3]})
        self.assertEqual(preempts, [])

    def test_waiting_gang_preempts_backfilled_evals(self):
        running = {"eval1": {"gpus": [1, 3], "priority": 10, "preemptible": True}}
        starts, preempts = decide([0, 2], running, [TRAIN])
        self.assertEqual((starts, preempts), ({}, ["eval1"]))

    def test_blocked_gang_keeps_non_preemptible_jobs_off(self):
        other = {"id": "sft1", "gpus": {"count": 1}, "priority": 10}
        starts, _ = decide([1], {"x": {"gpus": [0, 2, 3], "priority": 60}}, [TRAIN, other])
        self.assertEqual(starts, {})

    def test_elastic_takes_what_is_free_cpu_jobs_always_start(self):
        grade = {"id": "grade", "gpus": 0, "priority": 0}
        starts, _ = decide([0, 1, 2], {}, [EVAL(1), grade])
        self.assertEqual(starts, {"eval1": [0, 1, 2], "grade": []})

    def test_priority_order(self):
        hi = {"id": "hi", "gpus": {"count": 2}, "priority": 90}
        starts, _ = decide([0, 1], {}, [EVAL(1), hi])
        self.assertEqual(starts, {"hi": [0, 1]})


    def test_lent_gpus_only_for_preemptible_jobs(self):
        one = {"id": "merge", "gpus": {"count": 1}, "priority": 30}
        starts, _ = decide([6, 7], {}, [one, EVAL(1)], lent={6, 7})
        self.assertEqual(starts, {"eval1": [6, 7]})


if __name__ == "__main__":
    unittest.main()
