"""The pool daemon: every TICK seconds ingest submitted jobs, reap finished ones, decide (allocate),
preempt, launch. One per machine; state lives in results/pool/, so a restarted daemon picks up its
queue and re-attaches to jobs still running (they run in their own sessions).

GPUs: paths.SCHEDULED_GPUS, minus those another team is using (memory in use while we hold no
lease) and those leased by a legacy scheduler (runtime.scheduler.GpuLease) -- the daemon takes the
same leases for its jobs, so older tools and the bench service keep excluding them.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from mopd import paths
from mopd.pool import allocate, store
from mopd.runtime.gpus import gpu_memory_used
from mopd.runtime.scheduler import GpuLease

TICK = 5
LEND, RECALL = store.ROOT / "lend", store.ROOT / "recall"  # gpu<N> markers written by the bench service
STOP_GRACE = 60  # seconds between SIGTERM and SIGKILL when preempting
PREEMPTED = 75  # exit code of a job that stopped on request after saving its state (scripts/train_sft.py)
STOPS = store.ROOT / "stop"  # $MOPD_POOL_STOP files: a graceful job saves its state and exits when its file appears


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Daemon:
    def __init__(self, gpus=None):
        self.gpus = list(paths.SCHEDULED_GPUS if gpus is None else gpus)
        self.jobs = store.load_all()
        self.lease = GpuLease()
        self.procs = {}  # job id -> Popen (jobs launched by this daemon process)
        for j in self.jobs.values():  # re-attach: leases of jobs still running
            if j["state"] == "running" and alive(j.get("pid", -1)):
                self.lease.try_take(j["assigned"])
            elif j["state"] == "running":
                self.finish(j, None)

    def log(self, event, **fields):
        store.ROOT.mkdir(parents=True, exist_ok=True)
        with (store.ROOT / "events.jsonl").open("a") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **fields}) + "\n")

    # ---------------------------------------------------------------- jobs in and out
    def ingest(self):
        store.INBOX.mkdir(parents=True, exist_ok=True)
        for f in sorted(store.INBOX.glob("*.json")):
            msg = json.loads(f.read_text())
            f.unlink()
            if "cancel" in msg:
                j = self.jobs.get(msg["cancel"])
                if j and j["state"] == "running":
                    j["cancelled"] = True
                    self.stop(j)
                elif j and j["state"] == "queued":
                    self.set(j, state="cancelled")
                continue
            old = self.jobs.get(msg["id"])
            if old and old["state"] in ("queued", "running", "done"):
                continue  # resubmitting a live or finished job is a no-op
            job = {"priority": 0, "preemptible": False, "deps": [], "done": None, "gpus": 0, **msg,
                   "state": "queued", "submitted": time.time(), "attempts": 0}
            self.jobs[job["id"]] = job
            store.save(job)
            self.log("submit", job=job["id"])

    def set(self, job, **fields):
        job.update(fields)
        store.save(job)

    def finish(self, job, rc):
        held = job.get("assigned") or []  # before the record forgets them
        (STOPS / store.fname(job["id"])).unlink(missing_ok=True)
        if (job.get("stopping") or rc == PREEMPTED) and not job.get("cancelled"):
            self.set(job, state="queued", pid=None, assigned=[], stopping=False)  # preempted: runs again later
            self.log("preempted", job=job["id"])
        elif job.get("cancelled"):
            self.set(job, state="cancelled", pid=None, finished=time.time())
        elif rc == 0 or (rc is None and store.is_done(job["done"])):
            ok = not job["done"] or store.is_done(job["done"])
            self.set(job, state="done" if ok else "failed", pid=None, finished=time.time(),
                     **({} if ok else {"error": "exited 0 but its done check is false"}))
        else:
            self.set(job, state="failed", pid=None, finished=time.time(), error=f"exit code {rc}; see {job['log']}")
        self.lease.release(held)
        self.log("finish", job=job["id"], state=job["state"])

    def reap(self):
        for j in list(self.jobs.values()):
            if j["state"] != "running":
                continue
            p = self.procs.get(j["id"])
            rc = p.poll() if p else (None if alive(j["pid"]) else "gone")
            if rc is None:
                if j.get("stopping"):
                    waited = time.time() - j["stopping"]
                    grace = j.get("stop_grace", STOP_GRACE)
                    if j.get("graceful") and waited > grace and not j.get("termed"):
                        self.set(j, termed=True)
                        self.signal(j, signal.SIGTERM)  # did not stop on request in time
                    elif waited > grace + (STOP_GRACE if j.get("graceful") else 0):
                        self.signal(j, signal.SIGKILL)
                continue
            self.procs.pop(j["id"], None)
            self.finish(j, None if rc == "gone" else rc)

    # ---------------------------------------------------------------- decisions
    def ready(self):
        out = []
        for j in sorted(self.jobs.values(), key=lambda j: j["submitted"]):
            if j["state"] != "queued":
                continue
            if store.is_done(j["done"]):
                self.set(j, state="done", finished=time.time(), note="already done")
                continue
            deps = [self.jobs.get(d) for d in j["deps"]]
            if any(d is None or d["state"] in ("failed", "blocked", "cancelled") for d in deps):
                self.set(j, state="blocked", note=f"dependency not done: {j['deps']}")
                continue
            if all(d["state"] == "done" for d in deps):
                out.append(j)
        return out

    def lent(self):
        """Service GPUs the bench service lent us (idle) and has not recalled."""
        return [g for g in paths.SERVICE_GPUS if (LEND / f"gpu{g}").exists() and not (RECALL / f"gpu{g}").exists()]

    def recall(self):
        """Preempt whatever runs on a GPU the service wants back."""
        for g in paths.SERVICE_GPUS:
            if (RECALL / f"gpu{g}").exists():
                for j in self.jobs.values():
                    if j["state"] == "running" and g in (j.get("assigned") or []):
                        self.stop(j)

    def free_gpus(self):
        lent = self.lent()
        used = gpu_memory_used(self.gpus + lent)
        ours = {g for j in self.jobs.values() if j["state"] == "running" for g in j.get("assigned", [])}
        free = []
        for g in self.gpus + lent:
            if g in ours:
                continue
            if not self.lease.try_take([g]):  # a legacy scheduler of ours holds it
                continue
            self.lease.release([g])
            if used.get(g, 0) > 1000:  # another team's process (we hold no lease on it)
                continue
            free.append(g)
        return free

    def tick(self):
        self.ingest()
        self.reap()
        ready = self.ready()
        running = {j["id"]: {"gpus": j["assigned"], "priority": j["priority"], "preemptible": j["preemptible"]}
                   for j in self.jobs.values() if j["state"] == "running" and not j.get("stopping")}
        self.recall()
        starts, preempts = allocate.decide(self.free_gpus(), running, ready, lent=self.lent())
        for jid in preempts:
            self.stop(self.jobs[jid])
        for jid, gpus in starts.items():
            self.launch(self.jobs[jid], gpus)
        self.snapshot()

    # ---------------------------------------------------------------- processes
    def launch(self, job, gpus):
        if gpus and not self.lease.try_take(gpus):
            return  # raced with a legacy scheduler; next tick
        spec = store.ROOT / "specs" / store.fname(job["id"])
        spec.parent.mkdir(parents=True, exist_ok=True)
        spec.write_text(json.dumps(job))
        src = str(Path(__file__).resolve().parents[2])  # the code this daemon runs, for mopd.pool.run
        env = {**os.environ, "PYTHONPATH": src, "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)),
               "MOPD_POOL_JOB": job["id"],  # tools that lease GPUs themselves (bench_mopd) skip it: the pool holds them
               "MOPD_POOL_STOP": str(STOPS / store.fname(job["id"]))}  # graceful jobs watch for this file
        STOPS.mkdir(parents=True, exist_ok=True)
        (STOPS / store.fname(job["id"])).unlink(missing_ok=True)
        if not gpus:
            env["CUDA_VISIBLE_DEVICES"] = ""
        p = subprocess.Popen([str(paths.TRAIN_PY), "-m", "mopd.pool.run", str(spec)], cwd=paths.ROOT, env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        self.procs[job["id"]] = p
        self.set(job, state="running", assigned=gpus, pid=p.pid, started=time.time(), attempts=job["attempts"] + 1)
        self.log("start", job=job["id"], gpus=gpus)

    def signal(self, job, sig):
        try:
            os.killpg(os.getpgid(job["pid"]), sig)
        except (ProcessLookupError, TypeError):
            pass

    def stop(self, job):
        """Preempt (or cancel): SIGTERM the job's process group, SIGKILL after STOP_GRACE."""
        if job["state"] == "running" and not job.get("stopping"):
            self.set(job, stopping=time.time(), termed=False)
            if job.get("graceful") and not job.get("cancelled"):
                STOPS.mkdir(parents=True, exist_ok=True)
                (STOPS / store.fname(job["id"])).write_text("stop")  # it saves its state and exits PREEMPTED
            else:
                self.signal(job, signal.SIGTERM)
            self.log("stop", job=job["id"], cancelled=bool(job.get("cancelled")))

    def snapshot(self):
        counts = {}
        for j in self.jobs.values():
            counts[j["state"]] = counts.get(j["state"], 0) + 1
        running = {j["id"]: j["assigned"] for j in self.jobs.values() if j["state"] == "running"}
        tmp = store.ROOT / "state.json.tmp"
        tmp.write_text(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "gpus": self.gpus, "counts": counts,
                                   "running": running}, indent=1))
        os.replace(tmp, store.ROOT / "state.json")

    def serve(self):
        self.log("daemon_start", gpus=self.gpus, pid=os.getpid())
        while True:
            try:
                self.tick()
            except Exception as e:  # one bad job record must not stop the pool
                self.log("tick_error", error=repr(e)[:500])
            time.sleep(TICK)


def main():
    lock = store.ROOT / "daemon.pid"
    store.ROOT.mkdir(parents=True, exist_ok=True)
    if lock.exists() and alive(int(lock.read_text() or 0)):
        sys.exit(f"pool daemon already running (pid {lock.read_text()})")
    lock.write_text(str(os.getpid()))
    Daemon().serve()
