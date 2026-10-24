"""Scheduler jobs (mopd.runtime.scheduler) shared by every orchestrator: scripts/run_sweep.py,
scripts/run_experiment.py, scripts/run_eval_suite.py.

An evaluation is two jobs: generation on however many GPUs the scheduler grants (vLLM rows go to
whichever GPU has room), then grading on the CPU. Grading runs every test case for minutes; as a
GPU-free job of its own it no longer holds GPUs that the next generation or training could use.
"""

import json
import os
import subprocess
from pathlib import Path

from mopd import paths
from mopd.evaluation.eval_commands import eval_commands, grade_command
from mopd.students.registry import is_sharded


def shell(cmds, log):
    """Run [(argv, extra_env), ...] in order from the project root, appending to log; raise on failure."""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        for cmd, extra in cmds:
            f.write(f"\n$ {' '.join(map(str, cmd))}\n")
            f.flush()
            subprocess.run(cmd, cwd=paths.ROOT, env={**os.environ, **extra}, stdout=f, stderr=subprocess.STDOUT, check=True)


def generated(out):
    """Every row generated (graded or not)."""
    out = Path(out)
    if (out / "summary.json").exists():
        return True
    p = out / "generation-progress.json"
    if not p.exists():
        return False
    g = json.loads(p.read_text())
    return bool(g.get("total")) and g.get("done") == g.get("total")


def graded(out):
    return (Path(out) / "summary.json").exists()


def eval_jobs(job_id, *, out, data, protocol_id, max_new_tokens, log, deps=(), order=5, **model):
    """[generation job, grading job] for one model on one benchmark. model: model_version= +
    checkpoint= (a training run's checkpoint), or model= (+ adapter=, model_version_id=, ...) for a
    raw model directory. out is where the results go (inside the run for checkpoints)."""
    common = dict(data=data, out=out, protocol_id=protocol_id, max_new_tokens=max_new_tokens, grade=False, **model)
    return [
        {
            "id": job_id,
            "deps": list(deps),
            "order": order,
            "gpus": {"min": 1},
            "done": lambda: generated(out),
            "run": lambda gpus: shell(eval_commands(**common, device=gpus), log),
        },
        {
            "id": f"grade|{job_id}",
            "deps": [job_id],
            "order": order + 2,
            "gpus": 0,
            "done": lambda: graded(out),
            "run": lambda gpus: shell([grade_command(data, out)], log),
        },
    ]


def training_resources(config):
    """GPU request and gate for a training job: a sharded (full-parameter) student needs at least
    4 GPUs and a collective-bandwidth check first; a LoRA student runs on any GPU count."""
    if is_sharded(config):
        return {"gpus": {"min": 4}, "gate": "collectives"}
    return {"gpus": {"min": 1}, "gate": None}


def checkpoint_eval_dir(model_version, checkpoint, protocol_id):
    """Where scripts/evaluate_model_version.py writes a checkpoint's evaluation by default."""
    return Path(model_version).parent / "evaluations" / checkpoint / protocol_id
