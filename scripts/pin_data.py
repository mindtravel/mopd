"""Fill a spec's data pins with the sha256 of the files on the server, instead of copying digests by hand.

Experiment specs (configs/experiments/*.json): every "data_pins" entry whose value is "TBD", plus any
pool train/holdout file or evaluation file that is not pinned yet. Sweep specs (configs/sweeps/*.json):
eval.data_sha256 and test.data_sha256. Runs locally; digests are computed on the server over ssh.
Existing pins are left alone unless --refresh, so a changed file is caught at run time, not re-pinned.

  python scripts/pin_data.py configs/experiments/full256.json [--host para5090] [--refresh]
"""

import argparse, json, subprocess
from pathlib import Path


def remote_sha256(host, files):
    if not files:
        return {}
    out = subprocess.run(["ssh", host, "sha256sum", *files], capture_output=True, text=True, check=True).stdout
    return {line.split(None, 1)[1].strip(): line.split()[0] for line in out.splitlines()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--host", default="para5090")
    ap.add_argument("--refresh", action="store_true", help="re-pin files that already have a digest")
    a = ap.parse_args()
    path = Path(a.spec)
    spec = json.loads(path.read_text())
    if "variants" in spec:  # sweep
        slots = [(spec[k], "data", "data_sha256") for k in ("eval", "test") if k in spec]
        want = [s[f] for s, f, d in slots if a.refresh or s.get(d) in (None, "TBD")]
        digests = remote_sha256(a.host, sorted(set(want)))
        for s, f, d in slots:
            if s[f] in digests:
                s[d] = digests[s[f]]
        indent = 1
    else:  # experiment
        pins = spec.setdefault("data_pins", {})
        files = {f for p in spec.get("pools", {}).values() for f in [p["train_file"], *p["holdout_files"]]}
        files |= {spec["evaluation"]["data"]} if "evaluation" in spec else set()
        files |= set(pins)
        want = [f for f in files if a.refresh or pins.get(f) in (None, "TBD")]
        digests = remote_sha256(a.host, sorted(want))
        missing = sorted(set(want) - set(digests))
        if missing:
            raise SystemExit(f"not found on {a.host}: {missing}")
        pins.update(digests)
        indent = 2
    path.write_text(
        json.dumps(spec, indent=indent, ensure_ascii=False) + ("\n" if path.read_text().endswith("\n") else "")
    )
    print(json.dumps({"spec": str(path), "pinned": len(digests)}))


if __name__ == "__main__":
    main()
