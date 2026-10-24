"""Verify and record what a run is made of, before it trains a single step.

Every training entry calls record(config, run_dir) right after creating the run
directory. It refuses to start unless:
  - the deployed source is a clean commit (.deploy/source-version not "-dirty") and every
    deployed file still matches .deploy/manifest.sha256, i.e. nobody edited code on the
    server after scripts/dev/remote_sync.sh deployed it;
  - every data file pinned in config["data_pins"] has exactly the pinned sha256;
  - when the config was generated from an experiment spec (config["experiment"]), that
    spec file still has the recorded sha256, so all arms of one comparison provably
    share the controls the spec declares (scripts/make_experiment.py).
The result is written to <run>/provenance.json, next to config.json and the model-version
manifest, so any number traces back to code, data and spec.
"""

from mopd import paths
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

ROOT = paths.ROOT


class ProvenanceError(RuntimeError):
    """The run would not be reproducible from what is recorded; refuse to start."""


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source(root):
    version_file, manifest = root / ".deploy/source-version", root / ".deploy/manifest.sha256"
    if not version_file.is_file() or not manifest.is_file():
        raise ProvenanceError("no .deploy/source-version or manifest; deploy with scripts/dev/remote_sync.sh")
    version = version_file.read_text().strip()
    if "-dirty" in version:
        raise ProvenanceError(f"deployed source is not a clean commit ({version}); commit locally and re-sync")
    check = subprocess.run(["sha256sum", "--quiet", "-c", str(manifest)], cwd=root, capture_output=True, text=True)
    if check.returncode != 0:
        raise ProvenanceError(
            "deployed files differ from .deploy/manifest.sha256:\n" + (check.stdout + check.stderr)[-2000:]
        )
    commit = version.split("-", 1)[0]
    return {"source_version": version, "commit": commit, "deployed_files_verified": True}


def _data(config):
    pins = config.get("data_pins") or {}
    for path, expected in pins.items():
        actual = sha256(path)
        if actual != expected:
            raise ProvenanceError(
                f"data file changed since it was pinned: {path}\n  pinned {expected}\n  actual {actual}"
            )
    return {"data_pins_verified": sorted(pins)}


def _experiment(config):
    spec = config.get("experiment")
    if not spec:
        return {}
    actual = sha256(spec["spec_path"])
    if actual != spec["spec_sha256"]:
        raise ProvenanceError(f"experiment spec changed after configs were generated: {spec['spec_path']}")
    return {"experiment": spec}


def _versions():
    out = {"python": platform.python_version(), "host": platform.node()}
    for name in ("torch", "transformers", "peft", "vllm"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            pass
    return out


def record(config, run_dir, root=ROOT, filename="provenance.json"):
    record = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **_source(Path(root)),
        **_data(config),
        **_experiment(config),
        "versions": _versions(),
    }
    Path(run_dir, filename).write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return record
