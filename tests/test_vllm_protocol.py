import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "scripts")]
import unittest

from mopd.rollout.vllm_protocol import validate_item


class ValidateItemTest(unittest.TestCase):
    def setUp(self):
        self.source = {"index": 3, "prefix": [1, 2], "row": {"id": "x"}}

    def test_valid_answer_becomes_a_trajectory(self):
        t = validate_item({"answer": [5, 6, 99]}, self.source, 7, 4, 99)
        self.assertEqual((t["index"], t["answer"], t["student_version"]), (3, [5, 6, 99], 7))

    def test_rejects_empty_too_long_bad_tokens_and_tokens_after_eos(self):
        for answer in ([], [1, 2, 3, 4, 5], [1, -2], [1, 2.0], [99, 5]):
            with self.assertRaises(ValueError):
                validate_item({"answer": answer}, self.source, 7, 4, 99)


if __name__ == "__main__":
    unittest.main()
