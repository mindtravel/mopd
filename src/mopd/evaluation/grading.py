"""Official scorers, shared by benchmark grading and offline analysis."""

from mopd import paths
import os
import sys
from pathlib import Path

ROOT = paths.ROOT


def boxed(text):
    """Extract the last \\boxed{...} body; MATH answers must be explicitly final."""
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    begin = start + len(marker)
    depth = 1
    for index in range(begin, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
        if depth == 0:
            return text[begin:index]
    return None


def grade_math(solution, response):
    from math_verify import parse, verify

    gold, predicted = boxed(solution), boxed(response)
    if gold is None:
        return {"correct": False, "grade_error": "missing_gold_box"}
    if predicted is None:
        return {"correct": False, "grade_error": None, "prediction": None}
    try:
        parsed_gold = parse("$\\boxed{" + gold + "}$")
        parsed_predicted = parse("$\\boxed{" + predicted + "}$")
        return {
            "correct": bool(verify(parsed_gold, parsed_predicted)),
            "grade_error": None,
            "prediction": predicted,
            "gold": gold,
        }
    except Exception as error:
        return {"correct": False, "grade_error": repr(error), "prediction": predicted, "gold": gold}


def _instruction_lib():
    os.environ.setdefault("NLTK_DATA", str(ROOT / "cache/nltk"))
    vendor = str(ROOT / "vendor")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    from instruction_following_eval import evaluation_lib

    return evaluation_lib


def grade_instruction(row, text):
    """Official strict IFEval check for one response."""
    evaluation_lib = _instruction_lib()
    meta = row["metadata"]
    example = evaluation_lib.InputExample(
        key=meta["key"], instruction_id_list=meta["instruction_id_list"], prompt=row["prompt"], kwargs=meta["kwargs"]
    )
    result = evaluation_lib.test_instruction_following_strict(example, {row["prompt"]: text})
    return {
        "correct": bool(result.follow_all_instructions),
        "instruction_checks": [bool(v) for v in result.follow_instruction_list],
    }


def check(row, text):
    if row["domain"] == "math":
        return grade_math(row["solution"], text)
    return grade_instruction(row, text)


def summarize(results):
    """Per-domain accuracy, truncation and strict per-constraint accuracy."""
    summary = {}
    for domain in sorted({r["domain"] for r in results}):
        rows = [r for r in results if r["domain"] == domain]
        checks = [x for r in rows for x in r.get("instruction_checks", [])]
        summary[domain] = {
            "count": len(rows),
            "correct": sum(r["correct"] for r in rows),
            "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "truncated": sum(r["truncated"] for r in rows),
            "grade_errors": sum(bool(r.get("grade_error")) for r in rows),
        }
        if checks:
            summary[domain]["strict_instruction_accuracy"] = sum(checks) / len(checks)
    return summary
