"""Runs trained on an outdated data build: a training file's current sha256 differs from the pin in
the run's config (the file was rebuilt since). Such runs are not compared with anything; --archive
moves them under runs/archive/ (same relative path) and logs why in runs/archive/index.jsonl.
Studies never reuse them anyway: identities hash the pinned sha.

  python scripts/dev/stale_runs.py [--archive]        # on para5090
"""

import argparse, glob, hashlib, json, os, shutil, time

ROOT = "/ssd/lzx/mopd"
_sha = {}


def sha(path):
    if path not in _sha:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        _sha[path] = h.hexdigest()
    return _sha[path]


def stale_runs():
    patterns = ("runs/*/config.json", "runs/*/*/config.json", "runs/sweeps/*/*/config.json", "runs/studies/*/*/config.json")
    for cfg in sorted({p for pat in patterns for p in glob.glob(f"{ROOT}/{pat}")}):
        if "/archive/" in cfg or "/.stale" in cfg:
            continue
        try:
            c = json.load(open(cfg))
        except (OSError, ValueError):
            continue
        pins = c.get("data_pins", {})
        for s in (c.get("domains") or {}).values():
            f = s.get("train_file")
            if f and f in pins and os.path.exists(f) and sha(f) != pins[f]:
                yield os.path.dirname(cfg), f
                break


def referenced(run):
    """Code or configs naming this run's path (a hard-coded teacher broke the 2026-10-25 benchmark
    when its run was archived): archive only what nothing points at."""
    import subprocess

    rel = os.path.relpath(run, ROOT)
    out = subprocess.run(["grep", "-rlF", rel, f"{ROOT}/scripts", f"{ROOT}/src", f"{ROOT}/configs"], capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if "/experiments/" not in l]  # generated historical configs may name it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", action="store_true")
    a = ap.parse_args()
    for run, f in stale_runs():
        refs = referenced(run)
        print(run, f, f"REFERENCED BY {refs}" if refs else "")
        if a.archive and refs:
            print(f"  not archived: still referenced")
            continue
        if a.archive:
            rel = os.path.relpath(run, f"{ROOT}/runs")
            dest = f"{ROOT}/runs/archive/{rel}"
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.move(run, dest)
            with open(f"{ROOT}/runs/archive/index.jsonl", "a") as log:
                log.write(json.dumps({"run": run, "archived_to": dest, "train_file": f, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                      "reason": "trained on an outdated data build (train_file sha differs from the current file)"}) + "\n")


if __name__ == "__main__":
    main()
