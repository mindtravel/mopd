"""Checkpoint benchmark service on para5090: IFBench and AutomationBench (docker image mopd-bench,
docker/benchmarks) against a vLLM OpenAI server started on a free GPU for each job.

Checkpoints from other machines arrive through scripts/bench_client.py, which rsyncs them into
INBOX over ssh and calls this API through ssh as well; the service listens on 127.0.0.1 only, so the
shared server exposes no new port.

  POST /jobs        {"model": "/abs/path on this machine", "name": optional,
                     "benchmarks": ["ifbench", "automationbench"],
                     "ifbench": {"max_tokens": 4096, "workers": 64, "limit": null},
                     "automationbench": {"domains": "simple", "num_examples": null, "max_steps": 50,
                                         "max_concurrent": 32},
                     "max_model_len": 32768}
                    -> {"id": ...}
  GET  /jobs        all jobs (newest first)      GET /jobs/<id>   one job, with scores when done

A model directory with adapter_config.json is served as a LoRA on its base model. Each benchmark
owns one GPU (paths.SERVICE_GPUS: IFBench 6, AutomationBench 7; training and evals never get them)
and works through its own queue: a job's benchmarks run at the same time, each with its own vLLM.
The service holds both GPUs' leases while it runs.

  python scripts/bench_service.py [--port 18090]
"""

import argparse, json, os, shutil, signal, subprocess, sys, threading, time, traceback, urllib.request, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mopd import paths
from mopd.runtime import compile_cache
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
        self.lock, self.queues = threading.Lock(), {b: [] for b in BENCHMARKS}
        self.jobs = {}
        for f in sorted(self.dir.glob("*.json")):
            j = json.loads(f.read_text())
            self.jobs[j["id"]] = j
            for b, part in j.get("parts", {}).items():
                if part["state"] in ("queued", "running"):
                    part.update(state="queued", note="re-queued after service restart")
                    self.queues[b].append(j["id"])
            self.refresh(j)
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
            "ifbench": {"max_tokens": 4096, "workers": 64, "limit": None, **req.get("ifbench", {})},
            "automationbench": {
                "domains": "simple",
                "num_examples": None,
                "max_steps": 50,
                "max_concurrent": 32,
                **req.get("automationbench", {}),
            },
            "max_model_len": req.get("max_model_len", 32768),
            "parts": {b: {"state": "queued", "gpu": paths.SERVICE_GPUS[b]} for b in benches},
            "submitted": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with self.lock:
            self.jobs[j["id"]] = j
            for b in benches:
                self.queues[b].append(j["id"])
            self.refresh(j)
            self.save(j)
        return j

    def take(self, bench):
        with self.lock:
            return self.jobs[self.queues[bench].pop(0)] if self.queues[bench] else None

    @staticmethod
    def refresh(j):
        """Job-level state and scores from its parts."""
        states = {p["state"] for p in j["parts"].values()}
        j["state"] = ("failed" if "failed" in states else "complete") if states <= {"complete", "failed"} else (
            "running" if "running" in states or "complete" in states else "queued")
        j["scores"] = {b: p["scores"] for b, p in j["parts"].items() if p.get("scores")}
        j["errors"] = {b: p["error"] for b, p in j["parts"].items() if p.get("error")} or None

    def update(self, j, bench, **kw):
        with self.lock:
            j["parts"][bench].update(kw)
            self.refresh(j)
            self.save(j)


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
        "--gpu-memory-utilization", "0.9",
        "--max-num-seqs", "256",
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


def execute(jobs, j, bench):
    """One benchmark of one job on that benchmark's GPU: serve the model, run, record, stop vLLM."""
    gpu = paths.SERVICE_GPUS[bench]
    port = 18100 + gpu
    job_dir = STATE / "runs" / j["id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    log = job_dir / f"{bench}.log"
    proc = None
    try:
        jobs.update(j, bench, state="running", note="starting vLLM", started=time.strftime("%Y-%m-%dT%H:%M:%S"))
        vlog = job_dir / f"vllm-{bench}.log"
        # the GPU's own compiler caches (mopd.runtime.compile_cache): torch.compile / CUDA graphs are
        # reused by every later job on this GPU instead of being rebuilt per checkpoint
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
               **compile_cache.isolated(paths.ROOT / "cache/compile" / f"bench-gpu{gpu}")}
        with vlog.open("w") as f:
            proc = subprocess.Popen(vllm_command(j, port), env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        wait_ready(proc, port, vlog)
        jobs.update(j, bench, note="benchmark running")
        scores = RUNNERS[bench](j, job_dir, f"http://127.0.0.1:{port}/v1", log)
        jobs.update(j, bench, state="complete", scores=scores, note=None, finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    except Exception:
        jobs.update(j, bench, state="failed", error=traceback.format_exc()[-3000:], note=None,
                    finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    finally:
        if proc and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)


def worker(jobs, bench, lease):
    """Hold this benchmark's GPU lease for the service's lifetime (waiting out whoever has it now),
    then run its queue."""
    gpu = paths.SERVICE_GPUS[bench]
    while not lease.try_take([gpu]):
        jobs.waiting[bench] = f"GPU {gpu} still leased by an earlier job"
        time.sleep(30)
    while gpu_memory_used([gpu])[gpu] > 1000:
        jobs.waiting[bench] = f"GPU {gpu} still has memory in use"
        time.sleep(30)
    jobs.waiting.pop(bench, None)
    while True:
        j = jobs.take(bench)
        if j is None:
            time.sleep(5)
            continue
        execute(jobs, j, bench)


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
                return self.reply(200, {"ok": True, "gpus": paths.SERVICE_GPUS, "waiting": jobs.waiting,
                                        "queued": {b: len(q) for b, q in jobs.queues.items()}})
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
    a = ap.parse_args()
    if not shutil.which("docker"):
        raise SystemExit("docker not found")
    INBOX.mkdir(parents=True, exist_ok=True)
    jobs = Jobs()
    jobs.waiting = {}
    lease = GpuLease()
    for b in BENCHMARKS:
        threading.Thread(target=worker, args=(jobs, b, lease), daemon=True).start()
    print(f"bench service on 127.0.0.1:{a.port}, state {STATE}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), handler(jobs)).serve_forever()


if __name__ == "__main__":
    main()
