"""Streaming-pipeline pieces that need no GPU: batching, gradient attribution, stream validation."""
import random
import unittest

import torch

from mopd.rollout.base import RolloutStream, open_stream
from mopd.rollout.vllm_protocol import StreamValidator
from mopd.training.opd_step import GradContribution
from mopd.training.streaming import DomainBatcher


def traj(i, domain):
    return {"index": i, "row": {"domain": domain, "id": f"{domain}-{i}"}, "prefix": [1], "answer": [5, 2]}


class DomainBatcherTest(unittest.TestCase):
    def test_every_trajectory_once_grouped_and_sized(self):
        rows = [traj(i, d) for i, d in enumerate(["math"] * 11 + ["instruction"] * 11 + ["code"] * 10)]
        random.Random(0).shuffle(rows)  # arrival order
        out = list(DomainBatcher(4, {"math": 11, "instruction": 11, "code": 10}).consume(rows))
        seen = [t["index"] for _, chunk in out for t in chunk]
        self.assertEqual(sorted(seen), list(range(32)))
        for domain, chunk in out:
            self.assertTrue(all(t["row"]["domain"] == domain for t in chunk))
            self.assertLessEqual(len(chunk), 4)
        full = [len(c) for _, c in out]
        self.assertEqual(sum(1 for n in full if n < 4), 3)  # one remainder per domain (11, 11, 10 mod 4)

    def test_in_order_input_reproduces_whole_step_chunks(self):
        rows = [traj(i, "math" if i < 6 else "code") for i in range(10)]
        out = list(DomainBatcher(2, {"math": 6, "code": 4}).consume(rows))
        self.assertEqual([[t["index"] for t in c] for _, c in out], [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])

    def test_count_mismatch_and_unplanned_domain_fail(self):
        with self.assertRaises(ValueError):
            list(DomainBatcher(2, {"math": 3}).consume([traj(0, "math")]))
        with self.assertRaises(ValueError):
            list(DomainBatcher(2, {"math": 1}).consume([traj(0, "code")]))


class GradContributionTest(unittest.TestCase):
    def test_interleaved_attribution_matches_domain_major(self):
        torch.manual_seed(0)
        w = torch.randn(5, requires_grad=True)
        xs = {"a": [torch.randn(5) for _ in range(3)], "b": [torch.randn(5) for _ in range(2)]}
        # domain-major reference: grad after domain minus grad before domain
        ref = {}
        w.grad = None
        for d in ("a", "b"):
            before = w.grad.clone() if w.grad is not None else torch.zeros(5)
            for x in xs[d]:
                (w * x).sum().pow(2).backward()
            ref[d] = (w.grad - before).norm().item()
        total = w.grad.clone()
        # interleaved
        w.grad = None
        gc = GradContribution([w], ["a", "b"])
        for d, x in [("a", xs["a"][0]), ("b", xs["b"][0]), ("a", xs["a"][1]), ("b", xs["b"][1]), ("a", xs["a"][2])]:
            gc.before(); (w * x).sum().pow(2).backward(); gc.after(d)
        self.assertTrue(torch.allclose(w.grad, total))
        for d in ("a", "b"):
            self.assertAlmostEqual(gc.l2(d), ref[d], places=4)


class StreamValidatorTest(unittest.TestCase):
    def setUp(self):
        self.inputs = [{"index": 0, "prefix": [1], "row": {}}, {"index": 1, "prefix": [1], "row": {}}]
        self.v = StreamValidator(self.inputs, version=3, digest="x", max_new_tokens=8, eos=2)

    def msg(self, index, answer):
        return {"version": 3, "adapter_sha256": "x", "partial": {"index": index, "answer": answer}}

    def test_accepts_complete_stream(self):
        self.assertEqual(self.v.partial(self.msg(1, [4, 2]))["answer"], [4, 2])
        self.v.partial(self.msg(0, [5]))
        self.v.done({"version": 3, "adapter_sha256": "x", "done": True, "count": 2})

    def test_rejects_duplicate_stale_and_incomplete(self):
        self.v.partial(self.msg(0, [5]))
        with self.assertRaises(ValueError):
            self.v.partial(self.msg(0, [5]))
        with self.assertRaises(ValueError):
            self.v.partial({**self.msg(1, [5]), "version": 2})
        with self.assertRaises(ValueError):
            self.v.done({"version": 3, "adapter_sha256": "x", "done": True, "count": 1})


class OpenStreamTest(unittest.TestCase):
    def test_non_streaming_backend_is_wrapped(self):
        class Batch:
            def generate(self, student, rows, version):
                return [traj(0, "math")], 1.5, {"backend": "batch"}
        s = open_stream(Batch(), None, [], 0, streaming=True)
        self.assertEqual(len(list(s)), 1)
        self.assertGreaterEqual(s.seconds, 1.5)
        self.assertEqual(s.stats, {"backend": "batch"})


if __name__ == "__main__":
    unittest.main()
