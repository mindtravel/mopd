"""Checkpoint benchmark service on para5090: IFBench + AutomationBench against a resident vLLM on a
service GPU (paths.SERVICE_GPUS 6 and 7, equal workers on one queue; training never gets them).
A job takes one GPU and runs all its benchmarks at once on that engine, so their requests share
vLLM's batches; two checkpoints run side by side. Listens on 127.0.0.1 only: other machines use
scripts/bench_client.py over ssh. Each job records its engine's utilization (mopd.profiling) from
two /metrics reads, at no cost to the evaluation.

  POST /jobs   {"model": "/abs/path", "name"?, "benchmarks"?, "ifbench_limit"?, "auto_domains"?, "auto_limit"?}
  GET  /jobs   all jobs, newest first          GET /jobs/<id>   one job          GET /health

  python scripts/bench_service.py [--port 18090]
"""

import argparse
import json
import shutil
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from mopd import paths
from mopd.profiling.model import ModelShape, infer_utilization
from mopd.runtime.gpus import gpu_memory_used
from mopd.runtime.scheduler import GpuLease
from mopd.serving.benchmarks import RUNNERS
from mopd.serving.engine import STATE, Engine, counters
from mopd.serving.jobs import Jobs

now = lambda: time.strftime("%Y-%m-%dT%H:%M:%S")


def utilization(j, gpu, before, after, seconds):
    """The engine's work during the job (both benchmarks share it), as MFU / weight-bandwidth / batch."""
    d = {k: (after[k] - before[k]) if None not in (after[k], before[k]) else None for k in after}
    prefilled = (d["prompt"] or 0) - (d["prefix_hits"] or 0)  # prefix-cache hits are not computed
    u = infer_utilization(ModelShape.of_checkpoint(j["model"]), seconds=seconds, gpus=1, generated=d["generated"] or 0,
                          prefilled=max(prefilled, 0), steps=d["steps"], peaks=None)
    return {**u, "prefix_hit": round((d["prefix_hits"] or 0) / max(d["prompt"] or 0, 1), 3),
            "preemptions": int(d["preemptions"] or 0), "seconds": round(seconds), "gpu": gpu}


def execute(jobs, j, engine):
    job_dir = STATE / "runs" / j["id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    todo = [b for b in j["benchmarks"] if j["parts"][b]["state"] != "complete"]
    for b in todo:
        jobs.update(j, b, state="running", note="loading model", started=now())
    try:
        served, how = engine.use(j)
        jobs.update(j, gpu=engine.gpu, engine=how)
        before, mark = counters(engine.port), time.time()
    except Exception:
        engine.stop()
        for b in todo:
            jobs.update(j, b, state="failed", error=traceback.format_exc()[-3000:], note=None, finished=now())
        return

    def one(b):
        try:
            jobs.update(j, b, note="benchmark running")
            t = time.time()
            scores = RUNNERS[b](j, job_dir, f"http://127.0.0.1:{engine.port}/v1", served)
            jobs.update(j, b, state="complete", scores=scores, note=None, seconds=round(time.time() - t), finished=now())
        except Exception:
            jobs.update(j, b, state="failed", error=traceback.format_exc()[-3000:], note=None, finished=now())

    threads = [threading.Thread(target=one, args=(b,)) for b in todo]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    try:
        jobs.update(j, utilization=utilization(j, engine.gpu, before, counters(engine.port), time.time() - mark))
    except Exception as e:  # measurement must never fail a finished evaluation
        jobs.update(j, utilization={"error": str(e)[-300:]})
    if j["state"] == "failed":
        engine.stop()  # unknown state after a failure: the next job starts clean


IDLE_LEND_SECONDS = 600  # an idle service GPU is lent to the GPU pool (preemptible jobs only) after this
LEND, RECALL = paths.ROOT / "results/pool/lend", paths.ROOT / "results/pool/recall"


def acquire(jobs, gpu, lease):
    """Take this GPU's lease and wait until it is empty (an earlier job or a pool job let go)."""
    while not lease.try_take([gpu]):
        jobs.waiting[gpu] = f"GPU {gpu} still leased (a pool job being preempted, or an earlier job)"
        time.sleep(5)
    while gpu_memory_used([gpu])[gpu] > 1000:
        jobs.waiting[gpu] = f"GPU {gpu} still has memory in use"
        time.sleep(5)
    jobs.waiting.pop(gpu, None)


def worker(jobs, gpu, lease):
    """Serve jobs on this GPU. Idle for IDLE_LEND_SECONDS: stop the engine, release the lease and lend
    the GPU to the pool (mopd.pool: preemptible jobs only). A job arriving while lent recalls it: the
    pool preempts what runs there (evaluations resume later), then the service takes it back."""
    for d in (LEND, RECALL):
        d.mkdir(parents=True, exist_ok=True)
        (d / f"gpu{gpu}").unlink(missing_ok=True)
    acquire(jobs, gpu, lease)
    engine, idle_since, lent = Engine(gpu), time.time(), False
    while True:
        j = jobs.take()
        if j is None:
            if not lent and time.time() - idle_since > IDLE_LEND_SECONDS:
                engine.stop()
                lease.release([gpu])
                (LEND / f"gpu{gpu}").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
                lent = True
                jobs.waiting[gpu] = "lent to the GPU pool while idle"
            time.sleep(5)
            continue
        if lent:
            (LEND / f"gpu{gpu}").unlink(missing_ok=True)
            (RECALL / f"gpu{gpu}").write_text(j["id"])
            acquire(jobs, gpu, lease)
            (RECALL / f"gpu{gpu}").unlink(missing_ok=True)
            lent = False
        execute(jobs, j, engine)
        idle_since = time.time()


def handler(jobs):
    class H(BaseHTTPRequestHandler):
        def reply(self, code, body):
            data = json.dumps(body, ensure_ascii=False, indent=1).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            parts = self.path.strip("/").split("/")
            if parts == ["jobs"]:
                return self.reply(200, sorted(jobs.jobs.values(), key=lambda j: j["id"], reverse=True))
            if len(parts) == 2 and parts[0] == "jobs" and parts[1] in jobs.jobs:
                return self.reply(200, jobs.jobs[parts[1]])
            if parts == ["health"]:
                return self.reply(200, {"ok": True, "gpus": list(paths.SERVICE_GPUS), "waiting": jobs.waiting, "queued": len(jobs.queue)})
            self.reply(404, {"error": "not found"})

        def do_POST(self):
            if self.path.strip("/") != "jobs":
                return self.reply(404, {"error": "not found"})
            try:
                self.reply(200, jobs.submit(json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))))
            except (ValueError, KeyError) as e:
                self.reply(400, {"error": str(e)})

        def log_message(self, fmt, *args):
            pass

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18090)
    a = ap.parse_args()
    if not shutil.which("docker"):
        raise SystemExit("docker not found")
    (STATE / "inbox").mkdir(parents=True, exist_ok=True)
    jobs, lease = Jobs(STATE), GpuLease()
    for g in paths.SERVICE_GPUS:
        threading.Thread(target=worker, args=(jobs, g, lease), daemon=True).start()
    print(f"bench service on 127.0.0.1:{a.port}, state {STATE}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), handler(jobs)).serve_forever()
