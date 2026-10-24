"""Single writer for a run's status.json, so state never goes stale."""

import json
from pathlib import Path

FINAL_STATES = {"complete", "failed", "stopped"}


def write_status(run_dir, state, **fields):
    path = Path(run_dir) / "status.json"
    payload = {"state": state, **fields}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def read_status(run_dir):
    path = Path(run_dir) / "status.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def is_final(run_dir):
    return read_status(run_dir).get("state") in FINAL_STATES
