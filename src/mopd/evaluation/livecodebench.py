"""LiveCodeBench code-generation grading, following the official harness (lcb_runner testing_util).

Two problem kinds:
  - stdin (AtCoder and similar): the program reads stdin and writes stdout. Each test runs in its
    own subprocess; outputs match when the stripped non-empty lines are equal, or else every
    whitespace token is equal as a string or as a Decimal (the official float-tolerant fallback).
  - functional (LeetCode): the reply defines `class Solution` with the starter method. One
    subprocess runs all tests: each input line is one JSON argument, the result must equal the
    JSON-decoded expected output (tuples compared as lists).
A problem passes only if every public and private test passes; grading stops at the first failure.

Hidden tests ship as base64(zlib(pickle(json string))). They are decoded once by
scripts/prepare_unified_eval_v2.py with an unpickler that rejects every global (the payload is a
plain str), and stored as gzip JSON next to the benchmark file; rows reference that file and its
sha256. Execution uses the same isolation as code_exec (separate `python -I`, rlimits, timeout).
"""

import gzip
import hashlib
import io
import json
import pickle
import subprocess
import sys
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path

from mopd.evaluation import code_exec

TEST_TIMEOUT_SECONDS = 6  # official per-test timeout
MEMORY_LIMIT_BYTES = 4 << 30

# The one LiveCodeBench prompt, verbatim from the official harness
# (LiveCodeBench/LiveCodeBench lcb_runner/prompts/code_generation.py, get_generic_question_template_answer;
# the DeepSeek-R1 / QwQ templates use the same body inside their chat wrappers). Frozen: every benchmark
# file and every training recipe renders through prompt(); tests/test_lcb_prompt.py pins the exact text.
# Starter code is shown in the Format section, and the prompt ends with "### Answer:", not with code to
# continue, so the model writes one complete program. Reasoning students get their opening (<think>)
# from the student prompt format (mopd.protocol plain_qa_think_v1), never from this text.
FORMATTING_MESSAGE_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
)
FORMATTING_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the "
    "sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, "
    "it reads the inputs, runs the algorithm and writes output to STDOUT."
)


def prompt(question, starter=""):
    text = f"### Question:\n{question}\n\n"
    if starter:
        text += f"### Format: {FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        text += f"```python\n{starter}\n```\n\n"
    else:
        text += f"### Format: {FORMATTING_WITHOUT_STARTER_CODE}\n"
        text += "```python\n# YOUR CODE HERE\n```\n\n"
    text += "### Answer: (use the provided format with backticks)\n\n"
    return text


HEADER = """import sys, os, io, re, math, string, random, bisect, heapq, itertools, functools, collections, operator, copy, decimal, fractions, statistics
from typing import *
from collections import *
from itertools import *
from functools import *
from heapq import *
from bisect import *
from math import *
sys.setrecursionlimit(1 << 25)
"""

_FUNCTIONAL_RUNNER = """
import json, signal, sys
def _alarm(*_): raise TimeoutError()
signal.signal(signal.SIGALRM, _alarm)
def _norm(x):
    if isinstance(x, tuple): x = list(x)
    if isinstance(x, list): return [_norm(v) for v in x]
    return x
_tests = json.load(open(sys.argv[1]))
for _i, _t in enumerate(_tests):
    _args = [json.loads(line) for line in _t["input"].split("\\n")]
    signal.alarm({timeout})
    try:
        _got = getattr(Solution(), {func!r})(*_args)
    finally:
        signal.alarm(0)
    if _norm(_got) != _norm(json.loads(_t["output"])):
        print("FAIL", _i); sys.exit(1)
print("PASS")
"""


class _NoGlobals(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing global {module}.{name} in test payload")


def decode_private_tests(blob):
    import base64
    import zlib

    raw = _NoGlobals(io.BytesIO(zlib.decompress(base64.b64decode(blob)))).load()
    if not isinstance(raw, str):
        raise ValueError(f"expected a JSON string payload, got {type(raw).__name__}")
    return json.loads(raw)


def benchmark_rows(files, tests_dir, root, tag):
    """Benchmark rows from LCB release files; tests are written to tests_dir/<question_id>.json.gz."""
    for f in files:
        for line in Path(f).open():
            r = json.loads(line)
            tests = json.loads(r["public_test_cases"]) + decode_private_tests(r["private_test_cases"])
            kinds = {t["testtype"] for t in tests}
            if len(kinds) != 1:
                raise ValueError(f"{r['question_id']}: mixed test types {kinds}")
            kind = kinds.pop()
            path = Path(tests_dir) / f"{r['question_id']}.json.gz"
            path.write_bytes(gzip.compress(json.dumps(tests).encode(), mtime=0))
            meta = {
                "benchmark": "livecodebench",
                "testtype": kind,
                "tests_path": str(path.relative_to(root)),
                "tests_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "num_tests": len(tests),
                "difficulty": r["difficulty"],
                "platform": r["platform"],
                "contest_date": r["contest_date"],
                "release_file": Path(f).name,
            }
            starter = ""
            if kind == "functional":
                meta["func_name"] = json.loads(r["metadata"])["func_name"]
                starter = r["starter_code"]
            yield {
                "id": f"lcb/{r['question_id']}",
                "domain": "code",
                "split": "eval",
                "source": "livecodebench/code_generation_lite",
                "prompt": prompt(r["question_content"], starter),
                "problem": r["question_content"],
                "metadata": meta,
                "benchmark": tag,
            }


def load_tests(metadata, root):
    path = Path(root) / metadata["tests_path"]
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != metadata["tests_sha256"]:
        raise ValueError(f"test file digest mismatch: {path}")
    return json.loads(gzip.decompress(data))


def outputs_match(got, expected):
    a = [l.strip() for l in got.strip().splitlines() if l.strip()]
    b = [l.strip() for l in expected.strip().splitlines() if l.strip()]
    if a == b:
        return True
    ta, tb = " ".join(a).split(), " ".join(b).split()
    if len(ta) != len(tb):
        return False
    for x, y in zip(ta, tb):
        if x == y:
            continue
        try:
            if Decimal(x) != Decimal(y):
                return False
        except InvalidOperation:
            return False
    return True


def _limits():
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))


def _run_stdin(code, tests):
    with tempfile.TemporaryDirectory(prefix="mopd-lcb-") as tmp:
        path = Path(tmp) / "main.py"
        path.write_text(HEADER + "\n" + code)
        for i, t in enumerate(tests):
            try:
                r = subprocess.run(
                    [sys.executable, "-I", str(path)],
                    cwd=tmp,
                    input=t["input"],
                    capture_output=True,
                    text=True,
                    timeout=TEST_TIMEOUT_SECONDS,
                    preexec_fn=_limits,
                )
            except subprocess.TimeoutExpired:
                return {"passed": False, "error": f"test {i}: timeout"}
            if r.returncode != 0:
                return {"passed": False, "error": f"test {i}: " + r.stderr[-1500:]}
            if not outputs_match(r.stdout, t["output"]):
                return {"passed": False, "error": f"test {i}: wrong answer"}
    return {"passed": True, "error": None}


def _run_functional(code, tests, func):
    with tempfile.TemporaryDirectory(prefix="mopd-lcb-") as tmp:
        (Path(tmp) / "tests.json").write_text(json.dumps(tests))
        path = Path(tmp) / "main.py"
        path.write_text(
            HEADER + "\n" + code + "\n" + _FUNCTIONAL_RUNNER.format(timeout=TEST_TIMEOUT_SECONDS, func=func)
        )
        try:
            r = subprocess.run(
                [sys.executable, "-I", str(path), "tests.json"],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=TEST_TIMEOUT_SECONDS * len(tests) + 10,
                preexec_fn=_limits,
            )
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": "timeout"}
        if r.returncode == 0 and r.stdout.strip().endswith("PASS"):
            return {"passed": True, "error": None}
        return {"passed": False, "error": (r.stdout + r.stderr)[-1500:]}


def grade(row, text, root):
    """Same fields as code_exec.grade_code. Lenient adds only the format-tolerant extraction;
    the official LCB scorer takes the last fenced block, which is the strict score."""
    metadata = row["metadata"]
    tests = load_tests(metadata, root)

    def run(code):
        if not code.strip():
            return {"passed": False, "error": None}
        if metadata["testtype"] == "functional":
            return _run_functional(code, tests, metadata["func_name"])
        return _run_stdin(code, tests)

    strict_code = code_exec.extract_code(text)
    strict = run(strict_code)
    lenient_ok = strict["passed"]
    loose = code_exec.extract_code_lenient(text)
    if not lenient_ok and loose != strict_code:
        lenient_ok = run(loose)["passed"]
    return {
        "correct": strict["passed"],
        "correct_lenient": lenient_ok,
        "grade_error": None,
        "prediction": strict_code if strict_code.strip() else None,
        "execution_error": strict["error"],
    }
