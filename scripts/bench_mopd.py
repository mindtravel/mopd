"""Short full-parameter MOPD runs for infrastructure benchmarks (not experiments).

Takes an experiment arm's config, swaps in the MOPD study's teachers (configs/studies/mopd-mic.yaml), drops the experiment link (a benchmark
is not an arm, so provenance does not tie it to the spec), applies overrides, runs N steps through
torchrun, and prints the run's step summary (mopd.evaluation.report.opd_step_summary).

  python scripts/bench_mopd.py --name keepgather --steps 6 \
      --set 'fsdp={"reshard_after_forward": false}' --set micro_batch_tokens=16384 --set profile_step=3
"""

import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mopd import paths
from mopd.evaluation.report import opd_step_summary
from mopd.runtime.scheduler import GpuLease

TEACHER_STUDY = ROOT / "configs/studies/mopd-mic.yaml"  # teachers by reference (mopd.study), never by path


def teachers():
    """{domain: checkpoint directory} of the MOPD study's teachers, wherever those runs finished."""
    from mopd.study import runner, spec

    return runner.teacher_dirs(spec.load(TEACHER_STUDY), runner.finished_runs())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--arm", default="configs/experiments/full256-mopd/mopd-mic.json")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--set", action="append", default=[], help="KEY=JSON config override")
    a = ap.parse_args()
    c = json.loads((ROOT / a.arm).read_text())
    for d, path in teachers().items():
        if d in c["domains"]:
            c["domains"][d]["teacher"] = str(path)
    c.pop("experiment", None)
    out = paths.ROOT / "runs/scratch" / f"mopd-bench-{a.name}"
    c.update(steps=a.steps, save_every=10**6, output=str(out))
    for item in a.set:
        k, _, v = item.partition("=")
        c[k] = json.loads(v)
    cfg = paths.ROOT / "results" / f"scratch_mopd_bench_{a.name}.json"
    cfg.write_text(json.dumps(c, indent=1))
    shutil.rmtree(out, ignore_errors=True)
    n = c.get("student_gpus", 1)
    cmd = [
        str(paths.TRAIN_PY),
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={n}",
        "scripts/train_multidomain_opd.py",
        "--config",
        str(cfg),
    ]
    # hold the queues' GPU leases (mopd.runtime.scheduler.GpuLease) for every GPU the run touches, or a
    # queued eval can start on them mid-launch (2026-10-24: both OOM'd)
    ids = lambda ds: {int(str(d).split(":")[-1]) for d in ds}
    gpus = sorted(set(range(n)) | ids(c.get("rollout_devices", [])) | ids(d["device"] for d in c["domains"].values()))
    lease = GpuLease()
    pooled = bool(os.environ.get("MOPD_POOL_JOB"))  # launched by the pool: it holds the leases, GPUs are 0..n-1
    while not pooled and not lease.try_take(gpus):
        print(f"waiting for GPU leases {gpus}", flush=True)
        time.sleep(30)
    rc = subprocess.run(cmd, cwd=ROOT, env=os.environ).returncode
    if not pooled:
        lease.release(gpus)
    if (out / "metrics.jsonl").exists():
        print(json.dumps({"bench": a.name, **opd_step_summary(out / "metrics.jsonl")}))
    sys.exit(rc)


if __name__ == "__main__":
    main()
