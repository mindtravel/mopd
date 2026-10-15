"""Benchmark registry: where the data lives and which scorer applies."""
from pathlib import Path

from mopd.evaluation import grading

BENCHMARKS = {
    "ifeval": {
        "data": "data/instruction/ifeval.jsonl",
        "scorer": grading.grade_instruction,
        "domain": "instruction",
        "held_out": True,
    },
    "math-subset": {
        "data": "data/math/eval.jsonl",
        "scorer": grading.grade_math,
        "domain": "math",
        "held_out": True,
    },
    "all669": {
        "data": "data/full_eval/all.jsonl",
        "scorer": grading.check,
        "domain": None,
        "held_out": True,
    },
}


def get(name, project_root="/ssd/lzx/mopd"):
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark {name!r}; known: {', '.join(sorted(BENCHMARKS))}")
    spec = dict(BENCHMARKS[name])
    spec["path"] = Path(project_root) / spec["data"]
    return spec
