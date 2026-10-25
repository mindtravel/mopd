"""Studies (mopd.study): show the runs a study expands to, run it, or pin registry files.

  python scripts/study.py show configs/studies/full-sft-lr.yaml   # runs, identities, reused runs (on the server)
  python scripts/study.py submit configs/studies/full-sft-lr.yaml  # hand every job to the GPU pool (scripts/pool.py)
  python scripts/study.py run  configs/studies/full-sft-lr.yaml   # legacy: an in-process scheduler
  python scripts/study.py report configs/studies/full-sft-lr.yaml  # scores per run/suite/checkpoint vs baseline
  python scripts/study.py pin                                     # fill missing registry pins (sha256 over ssh)
"""

import argparse, json, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mopd import paths
from mopd.study import resolve, runner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["show", "run", "submit", "report", "pin"])
    ap.add_argument("--priority", type=int, default=0, help="submit: added to every job's priority")
    ap.add_argument("study", nargs="?")
    ap.add_argument("--host", default="para5090")
    a = ap.parse_args()
    if a.cmd == "pin":
        import yaml

        reg = resolve.registry()
        files = {e["data"] for e in reg["evals"].values()} | {d["train"] for d in reg["data"].values()}
        files |= {h for d in reg["data"].values() for h in d.get("holdout", [])}
        missing = sorted(f for f in files if f not in reg["pins"])
        if missing:
            out = subprocess.run(["ssh", a.host, "cd", str(paths.ROOT), "&&", "sha256sum", *missing],
                                 capture_output=True, text=True, check=True).stdout
            for line in out.splitlines():
                sha, f = line.split(None, 1)
                reg["pins"][f.strip()] = sha
            text = resolve.REGISTRY.read_text()
            head = text[: text.index("models:")]
            resolve.REGISTRY.write_text(head + yaml.safe_dump(reg, sort_keys=False, width=200))
        print(json.dumps({"pinned": len(missing)}))
        return
    if a.cmd == "submit":
        from mopd.pool import store

        specs = runner.pool_jobs(a.study, a.priority)
        store.submit(specs)
        print(json.dumps({"submitted": len(specs), "to": str(store.INBOX)}))
        return
    if a.cmd == "report":
        from mopd.study import report

        study, reg, runs = runner.plan(a.study)
        print(f"## {study['study']}\n\n" + report.markdown(report.table(study, reg, runs)))
        return
    if a.cmd == "show":
        study, _, runs = runner.plan(a.study)
        print(f"study {study['study']}: {len(runs)} runs")
        for r in runs:
            where = f"merge of {len(r['merge'])} experts -> {r['run_dir']}" if "merge" in r else f"reuse {r['run_dir']}" if r["reused"] else "train"
            print(f"  {r['name']:24s} {r['identity'] or '-':16s}  {where:60s} evals {','.join(r['evals'])}")
        return
    states = runner.run(a.study)
    failed = sorted(k for k, s in states.items() if s.get("state") in ("failed", "blocked"))
    print(json.dumps({"failed": failed}))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
