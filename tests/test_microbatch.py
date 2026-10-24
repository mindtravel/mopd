import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "scripts")]
import sys, unittest
from pathlib import Path
from mopd.training.microbatch import by_budget


class MicrobatchTest(unittest.TestCase):
    def test_count_cap_longest_first(self):
        batches = list(by_budget([3, 9, 1, 7, 5], lambda x: x, 2))
        self.assertEqual(batches, [[9, 7], [5, 3], [1]])

    def test_token_budget_runs_long_samples_alone(self):
        lengths = [10000, 400, 300, 300, 200]
        batches = list(by_budget(lengths, lambda x: x, max_count=8, max_tokens=1200))
        self.assertEqual(batches, [[10000], [400, 300, 300], [200]])  # 400*4=1600 > 1200 -> 200 starts a new one

    def test_every_item_exactly_once(self):
        items = list(range(1, 38))
        out = [x for b in by_budget(items, lambda x: x, 5, 60) for x in b]
        self.assertEqual(sorted(out), items)


if __name__ == "__main__":
    unittest.main()
