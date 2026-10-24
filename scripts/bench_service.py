"""Checkpoint benchmark service on para5090: IFBench and AutomationBench (docker image mopd-bench,
docker/benchmarks) against a vLLM OpenAI server started on a free GPU for each job.

Checkpoints from other machines arrive through scripts/bench_client.py, which rsyncs them into
INBOX over ssh and calls this API through ssh as well; the service listens on 127.0.0.1 only, so the
shared server exposes no new port.

  POST /jobs        {"model": "/abs/path on this machine", "name": optional,
                     "benchmarks": ["ifbench", "automationbench"],
                     "ifbench": {"max_tokens": 4096, "workers": 8, "limit": null},
                     "automationbench": {"domains": "simple", "num_examples": null, "max_steps": 50,
                                         "max_concurrent": 8},
                     "max_model_len": 32768}
                    -> {"id": ...}
  GET  /jobs        all jobs (newest first)      GET /jobs/<id>   one job, with scores when done

A model directory with adapter_config.json is served as a LoRA on its base model. GPUs are taken
through the queues' leases (mopd.runtime.scheduler.GpuLease), so jobs never land on a GPU that a
sweep or training holds; a job waits until some GPU is both unleased and empty.

  python scripts/bench_service.py [--port 18090] [--parallel 2]
"""

import argparse, json, os, shutil, signal, subprocess, sys, threading, time, traceback, urllib.request, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mopd import paths
from mopd.runtime.gpus import gpu_memory_used
from mopd.runtime.scheduler import GpuLease

STATE = paths.ROOT / "runs/bench-service"
INBOX = STATE / "inbox"
IMAGE = "mopd-bench:latest"
BENCHMARKS = ("ifbench", "automationbench")


class Jobs:
    """Job records, one JSON file each under STATE/jobs; a restart re-queues unfinished jobs."""

    def __init__(self):
        self.dir = STATE / "jobs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock, self.queue = threading.Lock(), []
        self.jobs = {}
        for f in sorted(self.dir.glob("*.json")):
            j = json.loads(f.read_text())
            self.jobs[j["id"]] = j
            if j["state"] in ("queued", "starting", "running"):
                j.update(state="queued", note="re-queued after service restart")
                self.queue.append(j["id"])
                self.save(j)

    def save(self, j):
        tmp = self.dir / f"{j['id']}.json.tmp"
        tmp.write_text(json.dumps(j, indent=1, ensure_ascii=False))
        os.replace(tmp, self.dir / f"{j['id']}.json")

    def submit(self, req):
        model = Path(req["model"])
        if not model.is_absolute() or not (model / "config.json").exists() and not (model / "adapter_config.json").exists():
            raise ValueError(f"{model}: not a model or LoRA adapter directory on this machine")
        benches = req.get("benchmarks", list(BENCHMARKS))
        if not benches or set(benches) - set(BENCHMARKS):
            raise ValueError(f"benchmarks must be a non-empty subset of {BENCHMARKS}")
        name = req.get("name") or model.name
        j = {
            "id": time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6],
            "name": name,
            "model": str(model),
            "benchmarks": benches,
            "ifbench": {"max_tokens": 4096, "workers": 8, "limit": None, **req.get("ifbench", {})},
            "automationbench": {
                "domains": "simple",
                "num_examples": None,
                "max_steps": 50,
                "max_concurrent": 8,
                **req.get("automationbench", {}),
            },
            "max_model_len": req.get("max_model_len", 32768),
            "state": "queued",
            "submitted": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with self.lock:
            self.jobs[j["id"]] = j
            self.queue.append(j["id"])
            self.save(j)
        return j

    def take(self):
        with self.lock:
            return self.jobs[self.queue.pop(0)] if self.queue else None

    def update(self, j, **kw):
        with self.lock:
            j.update(kw)
            self.save(j)


def free_gpu(lease):
    """Lease one GPU that no scheduler holds and nothing runs on; None if there is none now."""
    used = gpu_memory_used()
    for g in sorted(used, key=used.get):
        if used[g] < 1000 and lease.try_take([g]):
            if gpu_memory_used([g])[g] < 1000:  # recheck under the lease
                return g
            lease.release([g])
    return None


def vllm_command(j, port):
    model = Path(j["model"])
    cmd = [str(paths.INFER_PY.parent / "vllm"), "serve"]
    if (model / "adapter_config.json").exists():
        a = json.loads((model / "adapter_config.json").read_text())
        cmd += [a["base_model_name_or_path"], "--enable-lora", "--max-lora-rank", str(max(16, a.get("r", 16)))]
        cmd += ["--lora-modules", f"{j['name']}={model}"]
    else:
        cmd += [str(model), "--served-model-name", j["name"]]
    return cmd + [
        "--host", "127.0.0.1",
        "--port", str(port),
        "--dtype", "bfloat16",
        "--max-model-len", str(j["max_model_len"]),
        "--gpu-memory-utilization", "0.85",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
        "--seed", "0",
    ]


def wait_ready(proc, port, log, timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited ({proc.returncode}); tail of {log}:\n" + log.read_text()[-3000:])
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5).read()
            return
        except OSError:
            time.sleep(5)
    raise RuntimeError(f"vLLM not ready after {timeout} s")


def docker(job_dir, args, log):
    """Run the benchmark image against the host's vLLM (host network: bridge networking is broken here)."""
    cmd = ["docker", "run", "--rm", "--network", "host", "-e", "API_KEY=EMPTY", "-e", "OPENAI_API_KEY=EMPTY",
           "-v", f"{job_dir}:/out", "--name", f"mopd-bench-{job_dir.name}-{args[0]}", IMAGE, *args]
    with log.open("a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def run_ifbench(j, job_dir, api, log):
    o = j["ifbench"]
    gen = ["ifbench", "generate", "--run-name", "run", "--model", j["name"], "--api-base", api,
           "--workers", str(o["workers"]), "--max-tokens", str(o["max_tokens"]), "--temperature", "0"]
    if o.get("limit"):
        gen += ["--limit", str(o["limit"])]
    if docker(job_dir, gen, log) != 0:
        raise RuntimeError(f"ifbench generate failed; see {log}")
    if docker(job_dir, ["ifbench", "score", "--run-name", "run"], log) != 0:
        raise RuntimeError(f"ifbench score failed; see {log}")
    m = json.loads((job_dir / "ifbench/run/metrics.json").read_text())
    return {mode: {k: round(100 * v, 2) for k, v in m[mode].items() if "accuracy" in k} for mode in ("strict", "loose") if mode in m} or m


def run_automationbench(j, job_dir, api, log):
    o = j["automationbench"]
    args = ["automationbench", "--model", j["name"], "--base-url", api, "--api", "chat_completions",
            "--domains", o["domains"], "--max-steps", str(o["max_steps"]), "--max-concurrent", str(o["max_concurrent"]),
            "--export-json", "/out/automationbench/results.json"]
    if o.get("num_examples"):
        args += ["--num-examples", str(o["num_examples"])]
    if docker(job_dir, args, log) != 0:
        raise RuntimeError(f"automationbench failed; see {log}")
    return summarize_automation(json.loads((job_dir / "automationbench/results.json").read_text()))


def summarize_automation(r):
    """Pass rate (every assertion of a task passes) and mean partial score, overall and per domain."""
    rows = r.get("results") or r.get("rollouts") or r.get("samples") or []
    if not rows:
        return {"raw_keys": sorted(r)[:20]}
    by = {}
    for x in rows:
        d = x.get("domain") or x.get("task", {}).get("domain") or "all"
        by.setdefault(d, []).append(x)

    def agg(xs):
        rewards = [float(x.get("reward", x.get("score", 0)) or 0) for x in xs]
        return {"tasks": len(xs), "pass_rate": round(100 * sum(v >= 1 for v in rewards) / len(xs), 2),
                "mean_score": round(100 * sum(rewards) / len(xs), 2)}

    return {"all": agg(rows), **({d: agg(xs) for d, xs in by.items()} if len(by) > 1 else {})}


RUNNERS = {"ifbench": run_ifbench, "automationbench": run_automationbench}


def execute(jobs, j, lease):
    job_dir = STATE / "runs" / j["id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    log = job_dir / "service.log"
    jobs.update(j, state="starting", job_dir=str(job_dir), note="waiting for a free GPU")
    gpu = None
    while gpu is None:
        gpu = free_gpu(lease)
        if gpu is None:
            time.sleep(30)
    port = 18100 + gpu
    proc = None
    try:
        jobs.update(j, gpu=gpu, port=port, note="starting vLLM", started=time.strftime("%Y-%m-%dT%H:%M:%S"))
        vlog = job_dir / "vllm.log"
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "VLLM_USE_V1": "1"}
        with vlog.open("w") as f:
            proc = subprocess.Popen(vllm_command(j, port), env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        wait_ready(proc, port, vlog)
        api = f"http://127.0.0.1:{port}/v1"
        jobs.update(j, state="running", note="benchmarks running")
        scores, errors = {}, {}
        threads = []

        def one(b):
            try:
                scores[b] = RUNNERS[b](j, job_dir, api, log)
            except Exception as e:
                errors[b] = str(e)

        for b in j["benchmarks"]:  # concurrently: vLLM batches both clients' requests
            t = threading.Thread(target=one, args=(b,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        jobs.update(j, state="failed" if errors else "complete", scores=scores, errors=errors or None,
                    finished=time.strftime("%Y-%m-%dT%H:%M:%S"), note=None)
    except Exception:
        jobs.update(j, state="failed", errors={"service": traceback.format_exc()[-3000:]},
                    finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    finally:
        if proc and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
        lease.release([gpu])


def worker(jobs):
    lease = GpuLease()
    while True:
        j = jobs.take()
        if j is None:
            time.sleep(5)
            continue
        execute(jobs, j, lease)


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
                return self.reply(200, {"ok": True, "queued": len(jobs.queue)})
            self.reply(404, {"error": "not found"})

        def do_POST(self):
            if self.path.strip("/") != "jobs":
                return self.reply(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                self.reply(200, jobs.submit(req))
            except (ValueError, KeyError) as e:
                self.reply(400, {"error": str(e)})

        def log_message(self, fmt, *args):
            pass

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18090)
    ap.add_argument("--parallel", type=int, default=2, help="jobs at once (one GPU each)")
    a = ap.parse_args()
    if not shutil.which("docker"):
        raise SystemExit("docker not found")
    INBOX.mkdir(parents=True, exist_ok=True)
    jobs = Jobs()
    for _ in range(a.parallel):
        threading.Thread(target=worker, args=(jobs,), daemon=True).start()
    print(f"bench service on 127.0.0.1:{a.port}, state {STATE}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), handler(jobs)).serve_forever()


if __name__ == "__main__":
    main()
