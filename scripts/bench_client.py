"""Submit a checkpoint to the para5090 bench service (scripts/bench_service.py) from any machine
with ssh access to it; standard library only, so it runs outside the mopd environments.

The checkpoint is rsynced into the service inbox unless it is already a path on the server
(--remote-path); the API is reached through ssh, since the service listens on 127.0.0.1 only.

  python bench_client.py submit /path/to/ckpt [--name my-run] [--host para5090] \\
      [--benchmarks ifbench automationbench] [--ifbench-limit 12] [--auto-domains simple] \\
      [--auto-limit 8] [--wait]
  The evaluation settings themselves are fixed by the service (mopd.serving.protocol).
  python bench_client.py status [JOB_ID] [--host para5090]
"""

import argparse, json, shlex, subprocess, sys, time
from pathlib import Path

INBOX = "/ssd/lzx/mopd/runs/bench-service/inbox"
PORT = 18090


def api(host, method, path, body=None):
    cmd = f"curl -sS -X {method} http://127.0.0.1:{PORT}{path}"
    if body is not None:
        cmd += " -H 'Content-Type: application/json' --data-binary @-"
    out = subprocess.run(["ssh", host, cmd], input=json.dumps(body) if body is not None else None,
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("model")
    s.add_argument("--remote-path", action="store_true", help="model is already a path on the server")
    s.add_argument("--name")
    s.add_argument("--benchmarks", nargs="+", default=["ifbench", "automationbench"])
    s.add_argument("--ifbench-limit", type=int, help="first N prompts (default all 300)")
    s.add_argument("--auto-domains", default="simple", help="simple | all | comma list (sales,hr,...)")
    s.add_argument("--auto-limit", type=int, help="first N tasks (default all)")
    s.add_argument("--wait", action="store_true")
    st = sub.add_parser("status")
    st.add_argument("job", nargs="?")
    for p in (s, st):
        p.add_argument("--host", default="para5090")
    a = ap.parse_args()

    if a.cmd == "status":
        r = api(a.host, "GET", f"/jobs/{a.job}" if a.job else "/jobs")
        print(json.dumps(r, indent=1, ensure_ascii=False))
        return

    name = a.name or Path(a.model.rstrip("/")).name
    if a.remote_path:
        model = a.model
    else:
        src = Path(a.model).expanduser().resolve()
        if not src.is_dir():
            sys.exit(f"{src}: not a directory")
        model = f"{INBOX}/{name}"
        subprocess.run(["ssh", a.host, f"mkdir -p {shlex.quote(model)}"], check=True)
        subprocess.run(["rsync", "-a", "--partial", "--info=progress2",
                        "--exclude", "optimizer*", "--exclude", "training_state.pt", "--exclude", "*.pt",
                        f"{src}/", f"{a.host}:{model}/"], check=True)
    body = {"model": model, "name": name, "benchmarks": a.benchmarks, "ifbench_limit": a.ifbench_limit,
            "auto_domains": a.auto_domains, "auto_limit": a.auto_limit}
    j = api(a.host, "POST", "/jobs", body)
    if "error" in j:
        sys.exit(j["error"])
    print(f"job {j['id']} queued")
    while a.wait:
        time.sleep(30)
        j = api(a.host, "GET", f"/jobs/{j['id']}")
        if j["state"] in ("complete", "failed"):
            print(json.dumps({k: j.get(k) for k in ("id", "state", "scores", "utilization", "errors")}, indent=1, ensure_ascii=False))
            sys.exit(0 if j["state"] == "complete" else 1)
        print(f"{time.strftime('%H:%M:%S')} {j['state']} {j.get('note') or ''}", flush=True)


if __name__ == "__main__":
    main()
