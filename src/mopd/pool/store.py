"""Job records of the pool: one JSON file per job under results/pool/jobs/, new jobs dropped as JSON
files into results/pool/inbox/ (submit) and picked up by the daemon.

A job:
  {"id": "full-sft-lr/train:math-lr2e-06",          unique; resubmitting an id leaves a live job alone
   "commands": [[argv, extra_env], ...],            run in order from paths.ROOT ("{gpus}" -> GPU ids)
   "log": "/abs/path.log",
   "gpus": 0 | {"count": 4} | {"min": 1, "max": 4},
   "priority": 50, "preemptible": false,
   "deps": [job ids that must be done first],
   "done": {"kind": ..., "path": ...} | null}       finished already? (then never run); see is_done
States: queued -> running -> done | failed; a preempted job goes back to queued; a job whose
dependency failed becomes blocked.
"""

import json
import os
import time
import uuid
from pathlib import Path

from mopd import paths

ROOT = paths.ROOT / "results/pool"
JOBS, INBOX = ROOT / "jobs", ROOT / "inbox"


def fname(job_id):
    return job_id.replace("/", "__").replace(":", "_") + ".json"


def save(job):
    JOBS.mkdir(parents=True, exist_ok=True)
    tmp = JOBS / (fname(job["id"]) + ".tmp")
    tmp.write_text(json.dumps(job, indent=1, ensure_ascii=False))
    os.replace(tmp, JOBS / fname(job["id"]))


def load_all():
    JOBS.mkdir(parents=True, exist_ok=True)
    return {j["id"]: j for j in (json.loads(f.read_text()) for f in sorted(JOBS.glob("*.json")))}


def submit(jobs):
    """Drop jobs into the inbox (the daemon ingests them on its next tick)."""
    INBOX.mkdir(parents=True, exist_ok=True)
    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"  # unique: same-second submissions must not overwrite
    for i, job in enumerate(jobs):
        for key in ("id", "commands"):
            if key not in job:
                raise ValueError(f"job without {key}: {job}")
        tmp = INBOX / f".{stamp}-{i:04d}.tmp"
        tmp.write_text(json.dumps(job, ensure_ascii=False))
        os.replace(tmp, INBOX / f"{stamp}-{i:04d}.json")


def is_done(check):
    """Declarative done checks (evaluated by the daemon, so jobs carry no code)."""
    if not check:
        return False
    p = Path(check["path"])
    kind = check["kind"]
    if kind == "status_complete":  # a training run directory
        try:
            return json.loads((p / "status.json").read_text()).get("state") == "complete"
        except (OSError, ValueError):
            return False
    if kind == "generated":  # an evaluation's rows all generated
        from mopd.evaluation.jobs import generated

        return generated(p)
    if kind == "file":
        return p.exists()
    raise ValueError(f"unknown done check {kind!r}")
