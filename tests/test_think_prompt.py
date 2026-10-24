import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "scripts")]
import sys, unittest
from pathlib import Path
from mopd.evaluation.livecodebench import prompt_think


class ThinkPromptTest(unittest.TestCase):
    def test_no_code_block_or_starter_is_opened(self):
        p = prompt_think("Q", "functional", "class Solution:\n    def f(self, s: str) -> int:\n        ")
        self.assertNotIn("```python\nclass", p)
        self.assertIn("must define `class Solution` with the method `def f(self, s: str) -> int`", p)

    def test_bare_function_starter(self):
        p = prompt_think("Q", "functional", "def count(nums: List[int]) -> int:\n    ")
        self.assertIn("must define the function `def count(nums: List[int]) -> int`", p)
        self.assertNotIn("class Solution", p)

    def test_stdin(self):
        self.assertIn("stdin", prompt_think("Q", "stdin"))
        self.assertIn("Think through the problem first", prompt_think("Q", "stdin"))


if __name__ == "__main__":
    unittest.main()
