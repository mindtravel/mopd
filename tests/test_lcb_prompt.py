"""Pins livecodebench.prompt to the official LiveCodeBench generic template, byte for byte.
If this fails, the prompt was edited: it is frozen (benchmark files and training recipes depend on it)."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "scripts")]
import unittest

from mopd.evaluation.livecodebench import prompt

FUNCTIONAL = (
    "### Question:\nQ\n\n"
    "### Format: You will use the following starter code to write the solution to the problem and enclose "
    "your code within delimiters.\n"
    "```python\nclass Solution:\n    def f(self, s: str) -> int:\n        \n```\n\n"
    "### Answer: (use the provided format with backticks)\n\n"
)
STDIN = (
    "### Question:\nQ\n\n"
    "### Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not directly "
    "test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python "
    "program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
    "```python\n# YOUR CODE HERE\n```\n\n"
    "### Answer: (use the provided format with backticks)\n\n"
)


class LcbPromptTest(unittest.TestCase):
    def test_functional_is_the_official_template(self):
        self.assertEqual(prompt("Q", "class Solution:\n    def f(self, s: str) -> int:\n        "), FUNCTIONAL)

    def test_stdin_is_the_official_template(self):
        self.assertEqual(prompt("Q"), STDIN)


if __name__ == "__main__":
    unittest.main()
