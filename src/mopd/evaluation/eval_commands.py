"""One place that builds "generate then grade" commands, used by every eval orchestrator
(run_experiment.py, run_sweep.py, run_eval_suite.py, run_eval.py). Before this, each
callsite hand-built the same two-branch logic (a trained checkpoint goes through
evaluate_model_version.py, which generates and grades in one call; a raw model directory -- base
or a merged model -- has no model-version.json, so it chains generate_benchmark_vllm.py and
grade_benchmark_outputs.py by hand) independently, and the vLLM scheduling flags were a
copy-pasted constant in each file.
"""

from mopd import paths
import sys
from pathlib import Path

ROOT = paths.ROOT
PY = sys.executable

# Whole benchmark in one continuous-batching pass. Verified against the earlier 64-row batches on
# scale256 mix-mi step 256: cross-setting differences (581/1090 identical, 41 verdict flips) match
# same-setting repeats (589/38 and 594/40); generation wall time 482 s -> 155 s.
SCHEDULING = ["--batch-size", "4096", "--max-seqs", "256", "--max-num-batched-tokens", "8192"]


def eval_commands(
    *,
    data,
    protocol_id,
    max_new_tokens,
    device,
    student_prompt_format="plain_qa_v1",
    out=None,  # omitted for the model_version branch: evaluate_model_version.py
    # then writes inside the run (<run>/evaluations/<checkpoint>/<protocol>);
    # required for the raw-model branch, which has no run to write inside
    model=None,
    model_version_id=None,
    adapter=None,  # a raw model directory (base or merged)
    model_version=None,
    checkpoint=None,  # a training run's model-version.json + checkpoint
    max_model_len=None,
    scheduling=SCHEDULING,
    mopd_infer_python=None,
    temperature=0.0,
    top_p=1.0,
    grade=True,  # False: generation only (GPU); run grade_command() afterwards as a CPU job
):  # sampling: raw-model branch only for now
    """Returns [(argv, extra_env), ...] -- run each in order; the output directory ends up holding
    generated.jsonl, cases.jsonl and summary.json either way. Exactly one of (model_version,
    checkpoint) or (model,) must be given."""
    if bool(model_version) != bool(checkpoint):
        raise ValueError("model_version and checkpoint must be given together")
    if bool(model_version) == bool(model):
        raise ValueError("give exactly one of (model_version, checkpoint) or model")
    devices = [str(d) for d in (device if isinstance(device, (list, tuple)) else [device])]  # one GPU or several
    common = ["--data", str(data), "--protocol-id", protocol_id, "--max-new-tokens", str(max_new_tokens)]
    if max_model_len is not None:
        common += ["--max-model-len", str(max_model_len)]
    if model_version:
        if temperature:
            raise ValueError("sampling is only wired for the raw-model branch (model=..., adapter=...)")
        cmd = [
            PY,
            "scripts/evaluate_model_version.py",
            "--model-version",
            str(model_version),
            "--checkpoint",
            checkpoint,
            "--device",
            *devices,
            *common,
            *scheduling,
        ]
        if out is not None:
            cmd += ["--out", str(out)]
        if not grade:
            cmd += ["--skip-grade"]
        return [(cmd, {})]
    if out is None:
        raise ValueError("out is required when evaluating a raw model (no run to default into)")
    gen = [
        str(mopd_infer_python or paths.INFER_PY),
        "scripts/generate_benchmark_vllm.py",
        "--model",
        str(model),
        "--out",
        str(out),
        "--device",
        *devices,
        "--model-version-id",
        model_version_id or f"model:{Path(model).name}",
        "--student-prompt-format",
        student_prompt_format,
        *common,
        *scheduling,
    ]
    if adapter:
        gen += ["--adapter", str(adapter)]
    if temperature:
        gen += ["--temperature", str(temperature), "--top-p", str(top_p)]
    return [(gen, {})] + ([grade_command(data, out)] if grade else [])


def grade_command(data, out):
    """(argv, env) grading out/generated.jsonl on CPU into cases.jsonl + summary.json (idempotent)."""
    return [PY, "scripts/grade_benchmark_outputs.py", "--data", str(data), "--out", str(out)], {}
