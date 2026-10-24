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

A model directory with adapter_config.json is served as a LoRA on its base model. The service GPUs
(paths.SERVICE_GPUS, 6 and 7; training and evals never get them) are equal workers on one queue: a
job takes one GPU and runs all its benchmarks at once against that GPU's engine, so their requests
share vLLM's batches (IFBench's long-answer tail is filled with AutomationBench turns and vice versa)
and two checkpoints are evaluated side by side. AutomationBench runs as several client processes over
disjoint task ranges: one process cannot keep the engine busy. The service holds both GPUs' leases.

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
        self.lock, self.queue = threading.Lock(), []
        self.jobs = {}
        for f in sorted(self.dir.glob("*.json")):
            j = json.loads(f.read_text())
            self.jobs[j["id"]] = j
            if any(p["state"] in ("queued", "running") for p in j.get("parts", {}).values()):
                for p in j["parts"].values():
                    if p["state"] != "complete":
                        p.update(state="queued", note="re-queued after service restart")
                self.queue.append(j["id"])
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
                "max_concurrent": 8,  # per client process
                "shards": 4,  # client processes over disjoint task ranges
                **req.get("automationbench", {}),
            },
            "max_model_len": req.get("max_model_len", 32768),
            "parts": {b: {"state": "queued"} for b in benches},
            "submitted": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with self.lock:
            self.jobs[j["id"]] = j
            self.queue.append(j["id"])
            self.refresh(j)
            self.save(j)
        return j

    def take(self):
        with self.lock:
            return self.jobs[self.queue.pop(0)] if self.queue else None

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


def serving_signature(model):
    """What a running server must share with a checkpoint for an in-place weight swap: architecture
    shape, tokenizer and chat template (the server keeps its own tokenizer), and the LoRA base."""
    import hashlib

    model = Path(model)
    if (model / "adapter_config.json").exists():
        a = json.loads((model / "adapter_config.json").read_text())
        return {"lora_base": serving_signature(a["base_model_name_or_path"])}
    c = json.loads((model / "config.json").read_text())
    shape = {k: c.get(k) for k in ("architectures", "hidden_size", "num_hidden_layers", "num_attention_heads",
                                   "num_key_value_heads", "intermediate_size", "vocab_size", "tie_word_embeddings")}
    h = hashlib.sha256()
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
        if (model / f).exists():
            h.update(f.encode() + (model / f).read_bytes())
    return {**shape, "tokenizer": h.hexdigest()}


class Engine:
    """The resident vLLM OpenAI server on one service GPU. use(job) makes it serve the job's
    checkpoint and returns the model name to request: same signature -> weights loaded in place
    (WeightSwap, ~seconds) or a LoRA adapter added at runtime; otherwise a restart."""

    def __init__(self, gpu, extra=()):
        """extra: additional `vllm serve` arguments (profiling variants: KV dtype, batch budget)."""
        self.gpu, self.port, self.extra = gpu, 18100 + gpu, list(extra)
        self.proc = self.signature = None
        self.lora = set()

    def call(self, path, body, timeout=600):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=timeout).read().decode()

    def use(self, j, log):
        model = Path(j["model"])
        sig = {**serving_signature(model), "max_model_len": j["max_model_len"]}
        lora = (model / "adapter_config.json").exists()
        if self.proc and self.proc.poll() is None and sig == self.signature:
            mark = time.time()
            if lora:
                if j["name"] not in self.lora:
                    self.call("/v1/load_lora_adapter", {"lora_name": j["name"], "lora_path": str(model)})
                    self.lora.add(j["name"])
                how, served = "lora_added", j["name"]
            else:
                self.call("/collective_rpc", {"method": "load_checkpoint", "args": [str(model)]})
                self.call("/reset_prefix_cache", {})
                how, served = "weights_swapped", "policy"
            return served, {"engine": how, "seconds": round(time.time() - mark, 1)}
        self.stop()
        mark = time.time()
        cmd = [str(paths.INFER_PY.parent / "vllm"), "serve"]
        if lora:
            a = json.loads((model / "adapter_config.json").read_text())
            cmd += [a["base_model_name_or_path"], "--served-model-name", "policy", "--enable-lora",
                    "--max-lora-rank", str(max(64, a.get("r", 16))), "--max-loras", "4",
                    "--lora-modules", f"{j['name']}={model}"]
        else:
            cmd += [str(model), "--served-model-name", "policy"]
        cmd += ["--host", "127.0.0.1", "--port", str(self.port), "--dtype", "bfloat16",
                "--max-model-len", str(j["max_model_len"]), "--gpu-memory-utilization", "0.9",
                "--max-num-seqs", "256", "--enable-auto-tool-choice", "--tool-call-parser", "hermes", "--seed", "0",
                "--worker-extension-cls", "mopd.serving.weight_swap.WeightSwap", *self.extra]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(self.gpu),
               "VLLM_SERVER_DEV_MODE": "1",  # /collective_rpc, /reset_prefix_cache (server is on 127.0.0.1 only)
               "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1",
               "PYTHONPATH": str(paths.ROOT / "src"),
               # this GPU's compiler caches (mopd.runtime.compile_cache), reused across restarts
               **compile_cache.isolated(paths.ROOT / "cache/compile" / f"bench-gpu{self.gpu}")}
        self.vlog = STATE / f"vllm-gpu{self.gpu}.log"
        with self.vlog.open("a") as f:
            f.write(f"\n==== {time.strftime('%F %T')} {' '.join(cmd)}\n")
            f.flush()
            self.proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        wait_ready(self.proc, self.port, self.vlog)
        self.signature, self.lora = sig, ({j["name"]} if lora else set())
        return (j["name"] if lora else "policy"), {"engine": "started", "seconds": round(time.time() - mark, 1)}

    def stop(self):
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
        self.proc = self.signature = None


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


def docker(job_dir, args, log, name=None):
    """Run the benchmark image against the host's vLLM (host network: bridge networking is broken here)."""
    cmd = ["docker", "run", "--rm", "--network", "host", "-e", "API_KEY=EMPTY", "-e", "OPENAI_API_KEY=EMPTY",
           "-v", f"{job_dir}:/out", "--name", f"mopd-bench-{job_dir.name}-{name or args[0]}", IMAGE, *args]
    with log.open("a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def run_ifbench(j, job_dir, api, log, served):
    o = j["ifbench"]
    gen = ["ifbench", "generate", "--run-name", "run", "--model", served, "--api-base", api,
           "--workers", str(o["workers"]), "--max-tokens", str(o["max_tokens"]), "--temperature", "0"]
    if o.get("limit"):
        gen += ["--limit", str(o["limit"])]
    if docker(job_dir, gen, log) != 0:
        raise RuntimeError(f"ifbench generate failed; see {log}")
    if docker(job_dir, ["ifbench", "score", "--run-name", "run"], log) != 0:
        raise RuntimeError(f"ifbench score failed; see {log}")
    m = json.loads((job_dir / "ifbench/run/metrics.json").read_text())
    return {mode: {k: round(100 * v, 2) for k, v in m[mode].items() if "accuracy" in k} for mode in ("strict", "loose") if mode in m} or m


DOMAIN_TASKS = {"simple": 200, "sales": 100, "marketing": 100, "operations": 100, "support": 100, "finance": 100, "hr": 100}


def run_automationbench(j, job_dir, api, log, served):
    """Shards: contiguous task ranges (--skip/--num-examples) in parallel client processes, merged."""
    o = j["automationbench"]
    domains = list(DOMAIN_TASKS)[1:] if o["domains"] == "all" else o["domains"].split(",")
    total = o.get("num_examples") or sum(DOMAIN_TASKS[d] for d in domains)
    shards = max(1, min(o.get("shards", 1), total))
    bounds = [(k * total // shards, (k + 1) * total // shards) for k in range(shards)]
    codes = [None] * shards

    def shard(k):
        lo, hi = bounds[k]
        args = ["automationbench", "--model", served, "--base-url", api, "--api", "chat_completions",
                "--domains", o["domains"], "--max-steps", str(o["max_steps"]), "--max-concurrent", str(o["max_concurrent"]),
                "--skip", str(lo), "--num-examples", str(hi - lo), "--export-json", f"/out/automationbench/shard-{k}.json"]
        codes[k] = docker(job_dir, args, Path(f"{log}.shard{k}"), name=f"auto{k}")

    threads = [threading.Thread(target=shard, args=(k,)) for k in range(shards)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if any(codes):
        raise RuntimeError(f"automationbench shard(s) {[k for k, c in enumerate(codes) if c]} failed; see {log}.shard*")
    parts = [json.loads((job_dir / f"automationbench/shard-{k}.json").read_text()) for k in range(shards)]
    tasks = [t for p in parts for t in p["tasks"]]
    names = [t["name"] for t in tasks]
    if len(set(names)) != len(names) or len(tasks) != total:
        raise RuntimeError(f"automationbench shards returned {len(tasks)} tasks ({len(set(names))} distinct), expected {total}")
    merged = {"meta": {**parts[0]["meta"], "total_tasks": total, "shards": shards}, "tasks": tasks,
              "summary": {"pass_rate": sum(str(t["passed"]) == "True" for t in tasks) / total,
                          "avg_score": sum(float(t["score"]) for t in tasks) / total,
                          "total_tool_calls": sum(p["summary"].get("total_tool_calls") or 0 for p in parts),
                          "tasks_with_errors": sum(p["summary"].get("tasks_with_errors") or 0 for p in parts),
                          "aborted_tasks": [x for p in parts for x in p["summary"].get("aborted_tasks") or []]}}
    (job_dir / "automationbench/results.json").write_text(json.dumps(merged, ensure_ascii=False))
    return summarize_automation(merged)


def summarize_automation(r):
    """Pass rate (every assertion of a task passes) and mean partial score, overall and per domain
    (the task name's prefix: simple.*, sales.*, ...)."""
    def pct(v):
        return round(100 * float(v), 2)

    by = {}
    for t in r["tasks"]:
        by.setdefault(str(t["name"]).split(".")[0], []).append(t)
    out = {"all": {"tasks": len(r["tasks"]), "pass_rate": pct(r["summary"]["pass_rate"]), "mean_score": pct(r["summary"]["avg_score"]),
                   "tool_calls": r["summary"].get("total_tool_calls"), "errors": r["summary"].get("tasks_with_errors"),
                   "aborted": len(r["summary"].get("aborted_tasks") or [])}}
    if len(by) > 1:
        for d, ts in by.items():
            out[d] = {"tasks": len(ts), "pass_rate": pct(sum(str(t["passed"]) == "True" for t in ts) / len(ts)),
                      "mean_score": pct(sum(float(t["score"]) for t in ts) / len(ts))}
    return out


RUNNERS = {"ifbench": run_ifbench, "automationbench": run_automationbench}


def execute(jobs, j, engine):
    """All of a job's benchmarks at once on this worker's engine (shared batches)."""
    job_dir = STATE / "runs" / j["id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    now = lambda: time.strftime("%Y-%m-%dT%H:%M:%S")
    for b in j["benchmarks"]:
        if j["parts"][b]["state"] != "complete":
            jobs.update(j, b, state="running", note="loading model", gpu=engine.gpu, started=now())
    try:
        served, how = engine.use(j, job_dir / "service.log")
    except Exception:
        engine.stop()
        for b in j["benchmarks"]:
            jobs.update(j, b, state="failed", error=traceback.format_exc()[-3000:], note=None, finished=now())
        return

    def one(b):
        try:
            jobs.update(j, b, note="benchmark running", engine=how)
            mark = time.time()
            scores = RUNNERS[b](j, job_dir, f"http://127.0.0.1:{engine.port}/v1", job_dir / f"{b}.log", served)
            jobs.update(j, b, state="complete", scores=scores, note=None, bench_seconds=round(time.time() - mark), finished=now())
        except Exception:
            jobs.update(j, b, state="failed", error=traceback.format_exc()[-3000:], note=None, finished=now())

    threads = [threading.Thread(target=one, args=(b,)) for b in j["benchmarks"] if j["parts"][b]["state"] == "running"]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if j["state"] == "failed":
        engine.stop()  # unknown state after a failure: the next job starts clean


def worker(jobs, gpu, lease):
    """Hold this GPU's lease for the service's lifetime (waiting out whoever has it now), then take
    jobs from the shared queue."""
    while not lease.try_take([gpu]):
        jobs.waiting[gpu] = f"GPU {gpu} still leased by an earlier job"
        time.sleep(30)
    while gpu_memory_used([gpu])[gpu] > 1000:
        jobs.waiting[gpu] = f"GPU {gpu} still has memory in use"
        time.sleep(30)
    jobs.waiting.pop(gpu, None)
    engine = Engine(gpu)
    while True:
        j = jobs.take()
        if j is None:
            time.sleep(5)
            continue
        execute(jobs, j, engine)


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
                return self.reply(200, {"ok": True, "gpus": list(paths.SERVICE_GPUS), "waiting": jobs.waiting,
                                        "queued": len(jobs.queue)})
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
    for g in paths.SERVICE_GPUS:
        threading.Thread(target=worker, args=(jobs, g, lease), daemon=True).start()
    print(f"bench service on 127.0.0.1:{a.port}, state {STATE}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), handler(jobs)).serve_forever()


if __name__ == "__main__":
    main()
