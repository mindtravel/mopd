"""Execute generated code against HumanEval/MBPP tests in a throwaway subprocess.

Unlike math (regex/CAS comparison) or instruction (static constraint checks), code
correctness can only be judged by actually running the candidate function against its
test assertions. The candidate is model output, so it is never imported or exec'd in
this process: it is written to a fresh temp directory and run as a separate `python -I`
subprocess with a wall-clock timeout and POSIX resource limits (CPU time, address
space, no subprocesses), so a hang, a memory blowup, or an attempt to fork cannot
affect the evaluation process. This mirrors the standard HumanEval execution harness
(OpenAI's human-eval `execution.py`) rather than inventing a new sandboxing scheme.

This still only isolates the current process from a crashing or resource-hungry
candidate; it is not a security boundary against a deliberately adversarial payload
(no network isolation, no seccomp/namespace jail). That is an acceptable trade-off for
grading a fine-tuned 4B model's HumanEval/MBPP completions on a trusted research box,
not for running arbitrary or adversarial code.
"""
import re
import resource
import subprocess
import sys
import tempfile
from pathlib import Path

TIMEOUT_SECONDS = 10
MEMORY_LIMIT_BYTES = 1 << 30  # 1 GiB
CPU_TIME_LIMIT_SECONDS = 10

_CODE_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text):
    """Last fenced code block wins (same "last occurrence is final" convention as
    grading.boxed for math); falls back to the raw response if the model produced no
    fence at all, since a 4B student does not always fence a short completion."""
    matches = _CODE_FENCE.findall(text)
    if matches:
        return matches[-1].strip("\n")
    return text


def _limit_resources():
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_TIME_LIMIT_SECONDS, CPU_TIME_LIMIT_SECONDS))
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))  # no forking/subprocesses from the candidate


def _run_script(script):
    with tempfile.TemporaryDirectory(prefix="mopd-code-eval-") as tmp:
        path = Path(tmp) / "case.py"
        path.write_text(script)
        try:
            result = subprocess.run(
                [sys.executable, "-I", str(path)], cwd=tmp, timeout=TIMEOUT_SECONDS,
                capture_output=True, text=True, preexec_fn=_limit_resources,
            )
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": "timeout"}
        if result.returncode == 0:
            return {"passed": True, "error": None}
        return {"passed": False, "error": (result.stderr or result.stdout)[-2000:]}


def _humaneval_script(code, metadata):
    return f"{code}\n\n{metadata['test']}\n\ncheck({metadata['entry_point']})\n"


def _mbpp_script(code, metadata):
    imports = "\n".join(metadata.get("test_imports") or [])
    asserts = "\n".join(metadata["test_list"])
    return f"{imports}\n{code}\n\n{asserts}\n"


_BUILDERS = {"humaneval": _humaneval_script, "mbpp": _mbpp_script}


def grade_code(row, text):
    metadata = row["metadata"]
    benchmark = metadata["benchmark"]
    if benchmark not in _BUILDERS:
        raise ValueError(f"Unknown code benchmark {benchmark!r}")
    code = extract_code(text)
    if not code.strip():
        # No answer extracted, same as math's "predicted is None": a model miss, not a scorer failure.
        return {"correct": False, "grade_error": None, "prediction": None, "execution_error": None}
    script = _BUILDERS[benchmark](code, metadata)
    outcome = _run_script(script)
    # grade_error is reserved for the harness itself failing to produce a verdict; a failed
    # test is a normal incorrect answer (same as a wrong \boxed{} value for math), not that.
    return {"correct": outcome["passed"], "grade_error": None,
            "prediction": code, "execution_error": outcome["error"]}
