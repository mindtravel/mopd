"""One scheduler for every orchestrator (scripts/run_experiment.py, scripts/run_sweep.py): a job
DAG over a shared pool of GPUs, with fair dynamic GPU shares and a bandwidth gate for sharded
training.

A job is a dict:
  id      unique name
  deps    ids that must be complete first (a failed dependency blocks the job)
  gpus    0            CPU-only
          [0, 2, 3]    this fixed set, when all of it is free (MOPD's student/teacher/rollout layout)
          {"min": m, "max": M}   a fair share of the free GPUs, at least m, at most M (None = no cap):
                       share = max(m, free // jobs-not-holding-GPUs) -- a lone job takes every idle
                       GPU, concurrent jobs split them (GPUPool, below)
  gate    "collectives": before launching, time a model-sized all-gather on the granted GPUs
          (scripts/probe_collectives.py); if the host is too contended to train a sharded model
          at a useful speed, the GPUs are released, the job waits GATE_RETRY_SECONDS and other jobs
          proceed (2026-10-24: an external 64-core job made FSDP steps 60x slower).
  run     callable(gpus) that does the work and raises on failure
  done    callable() -> True when the job's output already exists (skipped, state "complete")
  order   smaller starts first among ready jobs (long trainings before evals)

Events go to <logdir>/events.jsonl, the live state to <logdir>/status.json, exactly as
run_experiment.py wrote them before.
"""

import json
import os
import subprocess
import threading
import time
from pathlib import Path

from mopd.evaluation.eval_commands import PY, ROOT

GATE_RETRY_SECONDS = 600
LEASE_DIR = Path("/dev/shm/mopd-gpu-leases")


class GpuLease:
    """Cross-process claim on GPUs: one flock'd file per GPU under LEASE_DIR, held while a job runs.
    Schedulers in different processes (two sweeps, a sweep and an experiment) otherwise only see
    memory use, and a GPU between an eval's generation and its CPU grading looks idle while another
    scheduler still owns it (2026-10-24: a training was handed such a GPU). try_take is all-or-none;
    the kernel drops the locks if the process dies."""

    def __init__(self):
        LEASE_DIR.mkdir(parents=True, exist_ok=True)
        self.files = {}

    def try_take(self, gpus):
        import fcntl

        taken = {}
        for g in gpus:
            f = open(LEASE_DIR / f"gpu{g}", "w")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                f.close()
                for h in taken.values():
                    h.close()
                return False
            taken[g] = f
        self.files.update(taken)
        return True

    def release(self, gpus):
        for g in gpus:
            f = self.files.pop(g, None)
            if f:
                f.close()


class GPUPool:
    """Shared GPUs handed out as a fair share of free // jobs-waiting-for-GPUs. The scheduler
    enqueues a job the moment it is ready (dependencies complete) and before its thread asks, so the
    denominator is exact and independent of thread arrival order: N jobs becoming ready together
    split the free GPUs N ways; a job that is the only one waiting takes them all; jobs still blocked
    behind a dependency are not counted."""

    def __init__(self, gpus):
        self._free = list(gpus)
        self._waiting = set()
        self._cv = threading.Condition()

    def enqueue(self, job_id):
        with self._cv:
            self._waiting.add(job_id)

    def want(self, job_id, min_gpus=1, max_gpus=None):
        with self._cv:
            self._waiting.add(job_id)
            while len(self._free) < min_gpus:
                self._cv.wait()
            share = min(max(min_gpus, len(self._free) // max(1, len(self._waiting))), len(self._free))
            if max_gpus is not None:
                share = min(share, max_gpus)
            taken, self._free = self._free[:share], self._free[share:]
            self._waiting.discard(job_id)
            return taken

    def want_exactly(self, job_id, gpus):
        """Block until this fixed set is entirely free, then take it."""
        with self._cv:
            self._waiting.add(job_id)
            while not set(gpus) <= set(self._free):
                self._cv.wait()
            self._free = [g for g in self._free if g not in gpus]
            self._waiting.discard(job_id)
            return list(gpus)

    def give_back(self, gpus):
        with self._cv:
            self._free.extend(gpus)
            self._cv.notify_all()


def collectives_ok(gpus, max_seconds=3.0):
    """(ok, seconds) of a model-sized bf16 all-gather across these GPUs; ok=True for a single GPU."""
    if len(gpus) < 2:
        return True, 0.0
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus))}
    r = subprocess.run(
        [
            PY,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={len(gpus)}",
            "scripts/probe_collectives.py",
            "--max-seconds",
            str(max_seconds),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    line = next((l for l in r.stdout.splitlines() if "all-gather" in l), r.stdout[-200:])
    try:
        seconds = float(line.split(":")[-1].split("s")[0])
    except ValueError:
        seconds = float("nan")
    return r.returncode == 0, seconds


class Scheduler:
    def __init__(self, gpus, logdir, min_free_gib=100.0, name="scheduler", poll_seconds=5.0):
        self.gpus, self.logdir, self.min_free, self.name = list(gpus), Path(logdir), min_free_gib, name
        self.poll_seconds = poll_seconds
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.jobs, self.state = {}, {}
        self.lock = threading.Lock()

    def add(self, job):
        job.setdefault("deps", [])
        job.setdefault("gpus", {"min": 1})
        job.setdefault("order", 5)
        job.setdefault("done", lambda: False)
        job.setdefault("gate", None)
        self.jobs[job["id"]] = job

    # ------------------------------------------------------------ bookkeeping
    def log(self, event, **fields):
        record = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "event": event,
            "ssd_free_gib": round(self.free_gib(), 1),
            **fields,
        }
        with self.lock, (self.logdir / "events.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    def save(self):
        with self.lock:
            (self.logdir / "status.json").write_text(json.dumps(self.state, indent=2))

    def free_gib(self):
        s = os.statvfs(self.logdir)  # the disk the run artifacts land on
        return s.f_bfree * s.f_frsize / 2**30

    # ------------------------------------------------------------ running
    def _acquire(self, job):
        spec = job["gpus"]
        if spec == 0:
            return []
        if isinstance(spec, list):
            return self.pool.want_exactly(job["id"], spec)
        return self.pool.want(job["id"], spec.get("min", 1), spec.get("max"))

    def _worker(self, job):
        while True:
            gpus = self._acquire(job)
            if gpus and not self.lease.try_take(gpus):  # another process's scheduler holds one of them
                self.pool.give_back(gpus)
                time.sleep(self.poll_seconds * 6)
                continue
            if job["gate"] == "collectives" and len(gpus) > 1:
                ok, seconds = collectives_ok(gpus)
                if not ok:
                    self.lease.release(gpus)
                    self.pool.give_back(gpus)
                    self.log(
                        "bandwidth_hold",
                        job=job["id"],
                        gpus=gpus,
                        all_gather_seconds=seconds,
                        retry_in=GATE_RETRY_SECONDS,
                    )
                    time.sleep(GATE_RETRY_SECONDS)
                    continue
            break
        self.state[job["id"]] = {"state": "running", "gpus": gpus, "started": time.strftime("%H:%M:%S")}
        self.save()
        self.log("start", job=job["id"], gpus=gpus)
        try:
            job["run"](gpus)
            self.state[job["id"]]["state"] = "complete"
        except Exception as error:
            self.state[job["id"]].update(state="failed", error=str(error)[-500:])
        finally:
            self.lease.release(gpus)
            self.pool.give_back(gpus)
        self.state[job["id"]]["finished"] = time.strftime("%H:%M:%S")
        self.save()
        self.log("finish", job=job["id"], state=self.state[job["id"]]["state"])

    def run(self):
        """Run every job once its dependencies are complete; returns the final state by job id."""
        for j in self.jobs.values():
            if j["done"]():
                self.state[j["id"]] = {"state": "complete", "skipped": True}
        self.save()
        self.pool = GPUPool(self.gpus)
        self.lease = GpuLease()
        threads = []
        while True:
            pending = [j for j in self.jobs.values() if j["id"] not in self.state]
            if not pending and all(s["state"] not in ("running", "starting") for s in self.state.values()):
                break
            for j in pending:
                failed = [d for d in j["deps"] if self.state.get(d, {}).get("state") in ("failed", "blocked")]
                if failed:
                    self.state[j["id"]] = {"state": "blocked", "by": failed}
            pending = [j for j in pending if j["id"] not in self.state]
            ready = sorted(
                (j for j in pending if all(self.state.get(d, {}).get("state") == "complete" for d in j["deps"])),
                key=lambda j: j["order"],
            )
            for j in ready:  # enqueue every ready job before any thread asks: exact fair shares
                if self.free_gib() < self.min_free:
                    self.log("low_disk_hold", job=j["id"])
                    break
                if j["gpus"] != 0:
                    self.pool.enqueue(j["id"])
                self.state[j["id"]] = {"state": "starting"}
            for j in ready:
                if self.state.get(j["id"], {}).get("state") == "starting" and not any(
                    t.name == j["id"] for t in threads
                ):
                    t = threading.Thread(target=self._worker, args=(j,), daemon=False, name=j["id"])
                    t.start()
                    threads.append(t)
            self.save()
            time.sleep(self.poll_seconds)
        for t in threads:
            t.join()
        counts = {}
        for s in self.state.values():
            counts[s["state"]] = counts.get(s["state"], 0) + 1
        self.log(f"{self.name}_done", **counts)
        return self.state
