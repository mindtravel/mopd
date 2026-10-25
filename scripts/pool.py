"""The GPU pool (mopd.pool): one daemon per machine runs every submitted job on matching GPUs.

  python scripts/pool.py serve                  # the daemon (para5090: setsid nohup ... &)
  python scripts/pool.py status [--all]         # GPUs, running and queued jobs
  python scripts/pool.py submit jobs.json       # a JSON list of job specs (mopd.pool.store)
  python scripts/pool.py cancel JOB_ID
"""

import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mopd.pool import store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["serve", "status", "submit", "cancel"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.cmd == "serve":
        from mopd.pool.daemon import main as serve

        serve()
    elif a.cmd == "submit":
        store.submit(json.loads(Path(a.arg).read_text()))
    elif a.cmd == "cancel":
        store.INBOX.mkdir(parents=True, exist_ok=True)
        (store.INBOX / f"{time.strftime('%Y%m%d-%H%M%S')}-{a.arg.replace('/', '_')}-cancel.json").write_text(json.dumps({"cancel": a.arg}))
    else:
        state = json.loads((store.ROOT / "state.json").read_text()) if (store.ROOT / "state.json").exists() else {}
        print(f"pool {state.get('time', '-')}: GPUs {state.get('gpus')}  {state.get('counts')}")
        jobs = sorted(store.load_all().values(), key=lambda j: (j["state"] != "running", -j["priority"], j["submitted"]))
        for j in jobs:
            if a.all or j["state"] in ("running", "queued", "failed", "blocked"):
                where = f"GPUs {j.get('assigned')}" if j["state"] == "running" else j.get("error") or j.get("note") or ""
                print(f"  {j['state']:9s} p{j['priority']:<3d} {j['id']:60s} {str(j['gpus']):28s} {where}"[:220])


if __name__ == "__main__":
    main()
