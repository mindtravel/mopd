"""Build HumanEval (164) + MBPP sanitized test split (257) as domain="code" benchmark
rows, same shape as data/full_eval/all.jsonl's math/instruction rows: {id, domain,
source, prompt, metadata}. The "prompt" field is the natural-language instruction
mopd.protocol.student_prompt wraps with "Instruction: ...\\nAnswer:" like any other
code-domain row (training and benchmark share one prompt protocol); metadata carries
whatever scripts/eval_domains.py's grade_code needs to actually run the tests.

Does not touch data/code/upstream/*: those parquet files were already downloaded by
scripts/download_code_pool.py for training-pool decontamination.
"""

import json
import re
from pathlib import Path
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
HUMANEVAL_INSTRUCTION = (
    "Complete the following Python function. Reply with the complete function "
    "(including the signature) in a single ```python code block and nothing else.\n\n{stub}"
)
MBPP_INSTRUCTION = (
    "Write a Python function for the following task. Reply with the complete function "
    "in a single ```python code block and nothing else.\n\n{task}\n\n"
    "Your function must pass tests such as:\n{example_test}"
)


def humaneval_rows():
    raw = pq.read_table(ROOT / "data/code/upstream/humaneval/test-00000-of-00001.parquet").to_pylist()
    for r in raw:
        yield {
            "id": r["task_id"],
            "domain": "code",
            "source": "openai/openai_humaneval",
            "prompt": HUMANEVAL_INSTRUCTION.format(stub=r["prompt"]),
            "metadata": {"benchmark": "humaneval", "entry_point": r["entry_point"], "test": r["test"]},
        }


def mbpp_rows():
    raw = pq.read_table(ROOT / "data/code/upstream/mbpp/sanitized-test-00000-of-00001.parquet").to_pylist()
    for r in raw:
        # The reference solution's own def line names the function more reliably than
        # parsing test_list[0], which sometimes wraps the call (e.g. math.isclose(f(...), ...)).
        match = re.search(r"^def\s+(\w+)\s*\(", r["code"], re.MULTILINE)
        if not match:
            raise ValueError(f"mbpp task {r['task_id']}: could not find a top-level def in the reference solution")
        yield {
            "id": f"mbpp/{r['task_id']}",
            "domain": "code",
            "source": "google-research-datasets/mbpp:sanitized",
            "prompt": MBPP_INSTRUCTION.format(task=r["prompt"], example_test=r["test_list"][0]),
            "metadata": {
                "benchmark": "mbpp",
                "entry_point": match.group(1),
                "test_list": r["test_list"],
                "test_imports": r["test_imports"],
            },
        }


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def main():
    out = ROOT / "data/code/benchmark"
    humaneval = list(humaneval_rows())
    mbpp = list(mbpp_rows())
    write(out / "humaneval.jsonl", humaneval)
    write(out / "mbpp.jsonl", mbpp)
    write(out / "all.jsonl", humaneval + mbpp)
    report = {
        "humaneval": len(humaneval),
        "mbpp": len(mbpp),
        "total": len(humaneval) + len(mbpp),
        "note": "MBPP uses the sanitized test split only (257 of 974 tasks); the sanitized "
        "prompt/train/validation splits were already consumed for decontamination, "
        "not benchmarking",
    }
    (out / "preparation_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
