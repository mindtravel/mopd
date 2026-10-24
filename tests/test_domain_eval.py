"""Validate wrapper behavior against official IFEval; no model calls."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "scripts")]
from eval_domains import check, summarize

r = {
    "id": "test",
    "domain": "instruction",
    "prompt": "Include apple.",
    "metadata": {"key": 0, "instruction_id_list": ["keywords:existence"], "kwargs": [{"keywords": ["apple"]}]},
}
assert check(r, "An apple.")["correct"]
assert not check(r, "An orange.")["correct"]
summary = summarize(
    [
        {"domain": "instruction", "correct": True, "truncated": False, "instruction_checks": [True, True]},
        {"domain": "instruction", "correct": False, "truncated": True, "instruction_checks": [False]},
    ]
)
assert summary["instruction"]["accuracy"] == 0.5
assert summary["instruction"]["strict_instruction_accuracy"] == 2 / 3
print("PASS: official strict keyword check and prompt/instruction aggregation")
