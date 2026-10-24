"""Streaming-pipeline pieces that need no GPU: batching, gradient attribution, stream validation."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "scripts")]
import random
import unittest

import torch

from mopd.rollout.base import RolloutStream, open_stream
from mopd.training.opd_step import BlockContribution
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


class BlockContributionTest(unittest.TestCase):
    def test_block_norms_are_each_domains_gradient(self):
        torch.manual_seed(0)
        w = torch.randn(5, requires_grad=True)
        xs = {"a": [torch.randn(5) for _ in range(3)], "b": [torch.randn(5) for _ in range(2)]}
        ref = {}
        for d in ("a", "b"):  # each domain's own gradient, computed alone
            w.grad = None
            for x in xs[d]:
                (w * x).sum().pow(2).backward()
            ref[d] = w.grad.norm().item()
        w.grad = None
        bc = BlockContribution([w], "cpu")
        for d in ("a", "b"):  # domains as consecutive blocks, gradients accumulating
            bc.before()
            for x in xs[d]:
                (w * x).sum().pow(2).backward()
            bc.after(d)
        for d in ("a", "b"):
            self.assertAlmostEqual(bc.l2(d), ref[d], places=4)


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
