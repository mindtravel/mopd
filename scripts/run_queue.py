"""Run a declared queue of experiment steps (configs/queues/<name>.json) through the shared scheduler,
instead of hand-written shell loops on the server.

  {"name": "...",
   "steps": [
     {"id": "...", "kind": "wait", "match": "run_sweep.py configs/sweeps/x.json"},  # an already-running process
     {"id": "...", "kind": "sweep", "spec": "configs/sweeps/x.json", "only": [...], "final": [...]},
     {"id": "...", "kind": "experiment", "plan": "configs/experiments/x/plan.json", "jobs": ["eval:expert-m:"]},
     {"id": "...", "kind": "eval", "model": "...", "adapter": null, "data": "...", "out": "...",
      "protocol_id": "...", "max_new_tokens": 8192, "student_prompt_format": "plain_qa_v1"},
     {"id": "...", "kind": "command", "argv": [..., "{gpus}", ...], "gpus": [0, 1, 2, 3, 4, 5, 6, 7] | 2 | 0}
   ]}

Steps run in order -- each after the previous one finishes, even if it failed -- unless a step lists
"deps" (ids that must succeed first). sweep/experiment steps start their own
orchestrator (which schedules its GPUs itself, under the machine-wide GPU leases); "command" steps
get the GPUs they ask for: a fixed set is leased and used by physical id by the command itself; a count
is passed as CUDA_VISIBLE_DEVICES, or as physical ids where the argv has "{gpus}"; "experiment" with
"jobs" keeps only plan jobs whose id starts with one of the prefixes (plus their dependencies).
A finished step leaves results/queues/<name>/<id>.done and is skipped on re-run; logs go to
logs/queues/<name>/<id>.log.

  python scripts/run_queue.py configs/queues/v3-rerun.json
"""

import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mopd import paths
from mopd.evaluation import jobs
from mopd.runtime.scheduler import Scheduler, usable_gpus


def running(match):
    """PIDs whose command line contains `match` (never this process or its ancestors)."""
    mine, pid = set(), os.getpid()
    while pid > 1:
        mine.add(pid)
        try:
            pid = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    found = []
    for d in Path("/proc").iterdir():
        if d.name.isdigit() and int(d.name) not in mine:
            try:
                cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            except OSError:
                continue
            if match in cmd:
                found.append(int(d.name))
    return found


def filtered_plan(plan_path, prefixes, out):
    plan = json.loads(Path(plan_path).read_text())
    by_id = {j["id"]: j for j in plan["jobs"]}
    keep = {i for i in by_id if any(i.startswith(p) for p in prefixes)}
    stack = list(keep)
    while stack:  # dependencies come along (they are skipped if already done)
        for d in by_id[stack.pop()]["deps"]:
            if d not in keep:
                keep.add(d)
                stack.append(d)
    plan["jobs"] = [j for j in plan["jobs"] if j["id"] in keep]
    Path(out).write_text(json.dumps(plan, indent=1))
    return out


def step_command(step, state_dir):
    py = str(paths.TRAIN_PY)
    k = step["kind"]
    if k == "sweep":
        argv = [py, "scripts/run_sweep.py", step["spec"]]
        if step.get("only"):
            argv += ["--only", *step["only"]]
        if step.get("final"):
            argv += ["--final", *step["final"]]
        return argv
    if k == "experiment":
        plan = step["plan"]
        if step.get("jobs"):
            plan = filtered_plan(plan, step["jobs"], state_dir / f"{step['id']}.plan.json")
        return [py, "scripts/run_experiment.py", str(plan)]
    if k == "command":
        return step["argv"]  # "{gpus}" is expanded by the caller
    raise ValueError(f"{step['id']}: unknown kind {k!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queue")
    a = ap.parse_args()
    q = json.loads(Path(a.queue).read_text())
    state_dir = paths.ROOT / "results/queues" / q["name"]
    logdir = paths.ROOT / "logs/queues" / q["name"]
    state_dir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)
    sched = Scheduler(usable_gpus(list(range(8))), state_dir, name=f"queue {q['name']}")
    prev = None
    for step in q["steps"]:
        sid = step["id"]
        # explicit deps must succeed; otherwise a step just runs after the previous one, even if it failed
        deps = step.get("deps", [])
        after = [] if "deps" in step or not prev else [prev]
        marker = state_dir / f"{sid}.done"
        log = logdir / f"{sid}.log"
        if step["kind"] == "wait":

            def wait(gpus, step=step):
                while running(step["match"]):
                    time.sleep(30)

            sched.add({"id": sid, "deps": deps, "after": after, "gpus": 0, "run": wait})
        elif step["kind"] == "eval":
            model = {k: step[k] for k in ("model", "adapter", "model_version_id", "student_prompt_format") if k in step}
            for job in jobs.eval_jobs(
                sid,
                out=step["out"],
                data=step["data"],
                protocol_id=step["protocol_id"],
                max_new_tokens=step["max_new_tokens"],
                log=log,
                deps=deps,
                after=after,
                **model,
            ):
                sched.add(job)
            sid = f"grade|{sid}"  # the next step waits for grading too
        else:
            gpus = step.get("gpus", 0)
            spec = gpus if isinstance(gpus, list) else ({"min": gpus, "max": gpus} if gpus else 0)

            def run(granted, step=step, log=log, marker=marker):
                argv = step_command(step, state_dir)
                # Fixed GPU sets are addressed by physical id inside the command's own config (MOPD
                # layouts: student/teacher/rollout devices), so the environment is left alone -- a
                # CUDA_VISIBLE_DEVICES subset would renumber them. A count gets CUDA_VISIBLE_DEVICES
                # unless the command takes the ids itself through "{gpus}".
                takes_ids = "{gpus}" in argv
                env = {}
                if granted and not isinstance(step.get("gpus"), list) and not takes_ids:
                    env = {"CUDA_VISIBLE_DEVICES": ",".join(map(str, granted))}
                argv = [x for arg in argv for x in ([str(g) for g in granted] if arg == "{gpus}" else [arg])]
                jobs.shell([(argv, env)], log)
                marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))

            sched.add({"id": sid, "deps": deps, "after": after, "gpus": spec, "done": marker.exists, "run": run})
        prev = sid
    state = sched.run()
    failed = sorted(k for k, s in state.items() if s["state"] in ("failed", "blocked"))
    if failed:
        raise SystemExit(f"failed steps: {failed}")


if __name__ == "__main__":
    main()
