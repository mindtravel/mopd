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


_CODE_START = re.compile(r"^(def |class |import |from \S+ import |@)")
_CODE_LINE = re.compile(r"^(\s|def |class |import |from |@|#|if __name__|[A-Za-z_][A-Za-z0-9_]*\s*=)")


def extract_code_lenient(text):
    """Format-tolerant extraction for the lenient metric only.

    With a fenced block this equals extract_code. Without one, keep the span from the
    first top-level def/class/import line up to the first unindented line that is not
    code (prose after the function), instead of executing the whole response verbatim.
    """
    if _CODE_FENCE.search(text):
        return extract_code(text)
    # plain_qa prompts end in "Answer:", so an unfenced reply often starts " def ...": shift every
    # line left by the first code line's own indentation (the body keeps its relative indentation).
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if _CODE_START.match(line.lstrip())), None)
    if start is None:
        return text
    offset = len(lines[start]) - len(lines[start].lstrip())
    kept = []
    for line in lines[start:]:
        line = line[offset:] if line[:offset].strip() == "" else line
        if line.strip() and not _CODE_LINE.match(line):
            break
        kept.append(line)
    return "\n".join(kept).rstrip() + "\n"


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


def _humaneval_stub(row):
    """The original HumanEval prompt (signature, docstring, helpers, imports) embedded in our
    instruction after the fixed preamble (scripts/prepare_code_benchmark.py)."""
    marker = "nothing else.\n\n"
    return row["prompt"].split(marker, 1)[1] if marker in row["prompt"] else None


def grade_code(row, text, lenient=True):
    """Strict fields as before; with lenient=True also `correct_lenient` (format tolerance):
    unfenced code is extracted heuristically, and for HumanEval the original prompt stub is
    prepended, as the official harness executes prompt + completion -- a reply that rewrites
    only the target function then still has the stub's helper functions and imports."""
    if row["metadata"]["benchmark"] == "livecodebench":
        from mopd.evaluation import livecodebench
        from mopd.evaluation.grading import ROOT
        return livecodebench.grade(row, text, ROOT)
    result = _grade_extracted(row, extract_code(text))
    if lenient:
        loose = extract_code_lenient(text)
        # Lenient = strict plus format tolerance, so it never drops below strict.
        ok = result["correct"] or (loose != result["prediction"] and _grade_extracted(row, loose)["correct"])
        stub = _humaneval_stub(row) if row["metadata"]["benchmark"] == "humaneval" else None
        if not ok and stub and loose.strip():
            ok = _grade_extracted(row, stub + "\n\n" + loose)["correct"]
        result["correct_lenient"] = ok
    return result


def _grade_extracted(row, code):
    metadata = row["metadata"]
    benchmark = metadata["benchmark"]
    if benchmark not in _BUILDERS:
        raise ValueError(f"Unknown code benchmark {benchmark!r}")
    if not code.strip():
        # No answer extracted, same as math's "predicted is None": a model miss, not a scorer failure.
        return {"correct": False, "grade_error": None, "prediction": None, "execution_error": None}
    script = _BUILDERS[benchmark](code, metadata)
    outcome = _run_script(script)
    # grade_error is reserved for the harness itself failing to produce a verdict; a failed
    # test is a normal incorrect answer (same as a wrong \boxed{} value for math), not that.
    return {"correct": outcome["passed"], "grade_error": None,
            "prediction": code, "execution_error": outcome["error"]}
