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

# Imports the official harness prepends to every candidate.
# Prompt wording follows the official LCB generation prompt, adapted to one instruction.
INSTRUCTION = ("You will be given a question (problem specification) and will generate a correct Python program "
               "that matches the specification and passes all tests.\n\nQuestion: {question}\n\n")
STDIN_FORMAT = ("Read the inputs from stdin, solve the problem and write the answer to stdout (do not directly test on "
                "the sample inputs). Reply with the complete program in a single ```python code block.")
FUNCTIONAL_FORMAT = ("Use the following starter code and reply with the complete solution in a single ```python code "
                     "block.\n```python\n{starter}\n```")


def prompt(question, kind, starter=""):
    return INSTRUCTION.format(question=question) + (FUNCTIONAL_FORMAT.format(starter=starter) if kind == "functional" else STDIN_FORMAT)


# "Think first" protocol (lcb-dev-v2-think / student prompt format plain_qa_think_v1): the prompt
# neither opens a code block nor shows starter code to continue; the student prompt ends by opening
# the reasoning (<think>), and the reply is reasoning, </think>, then one complete code block. The
# harness still calls Solution().<method>, so a functional problem states the required class and
# method signature in prose.
THINK_TAIL = ("Think through the problem first. Then give the complete {what} in a single ```python code block "
              "after your reasoning.")
THINK_STDIN = "Read the inputs from stdin, solve the problem and write the answer to stdout (do not test on the sample inputs). "
THINK_FUNCTIONAL = "Your solution must define `class Solution` with the method `{signature}`. "


def signature_of(starter):
    """`def name(self, ...) -> T` lines of a LeetCode starter, joined; the whole starter if none."""
    defs = [l.strip().rstrip(":") for l in starter.splitlines() if l.strip().startswith("def ")]
    return "; ".join(defs) if defs else starter.strip()


def prompt_think(question, kind, starter=""):
    lead = THINK_FUNCTIONAL.format(signature=signature_of(starter)) if kind == "functional" else THINK_STDIN
    return INSTRUCTION.format(question=question) + lead + THINK_TAIL.format(what="solution" if kind == "functional" else "program")


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


def benchmark_rows(files, tests_dir, root, tag, style="code"):
    """style "code": v1 prompts (prompt()); "think": prompt_think()."""
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
            meta = {"benchmark": "livecodebench", "testtype": kind, "tests_path": str(path.relative_to(root)),
                    "tests_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "num_tests": len(tests),
                    "difficulty": r["difficulty"], "platform": r["platform"], "contest_date": r["contest_date"],
                    "release_file": Path(f).name}
            starter = ""
            if kind == "functional":
                meta["func_name"] = json.loads(r["metadata"])["func_name"]
                starter = r["starter_code"]
            yield {"id": f"lcb/{r['question_id']}", "domain": "code", "split": "eval",
                   "source": "livecodebench/code_generation_lite",
                   "prompt": (prompt_think if style == "think" else prompt)(r["question_content"], kind, starter),
                   "metadata": meta, "benchmark": tag}


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
                r = subprocess.run([sys.executable, "-I", str(path)], cwd=tmp, input=t["input"], capture_output=True,
                                   text=True, timeout=TEST_TIMEOUT_SECONDS, preexec_fn=_limits)
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
        path.write_text(HEADER + "\n" + code + "\n" + _FUNCTIONAL_RUNNER.format(timeout=TEST_TIMEOUT_SECONDS, func=func))
        try:
            r = subprocess.run([sys.executable, "-I", str(path), "tests.json"], cwd=tmp, capture_output=True, text=True,
                               timeout=TEST_TIMEOUT_SECONDS * len(tests) + 10, preexec_fn=_limits)
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
    return {"correct": strict["passed"], "correct_lenient": lenient_ok, "grade_error": None,
            "prediction": strict_code if strict_code.strip() else None, "execution_error": strict["error"]}
