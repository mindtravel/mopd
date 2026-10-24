"""Short profiling runs of the bench service's serving path (scripts/bench_service.py): each variant
serves one model on a service GPU with given `vllm serve` arguments, runs a benchmark subset at a given
concurrency, and reports wall time with vLLM's own counters (token throughput, preemptions, prefix-cache
hits) and sampled queue depth / KV-cache usage / GPU utilization. The service must be stopped (it
holds GPUs 6/7); variants on the two GPUs run in parallel.

  python scripts/dev/profile_bench_service.py [--model ...] [--auto-tasks 32] [--if-prompts 64]
"""

import argparse, json, re, subprocess, sys, threading, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import bench_service as bs
from mopd import paths
from mopd.runtime.scheduler import GpuLease

FP8 = ["--kv-cache-dtype", "fp8"]


def metrics(port):
    text = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10).read().decode()
    out = {}
    for name in ("generation_tokens_total", "prompt_tokens_total", "num_preemptions_total", "prefix_cache_queries_total",
                 "prefix_cache_hits_total", "num_requests_running", "num_requests_waiting", "kv_cache_usage_perc"):
        m = re.search(rf"^vllm:{name}{{[^}}]*}} ([0-9.e+-]+)$", text, re.M)
        out[name] = float(m.group(1)) if m else 0.0
    return out


def gpu_util(gpu):
    out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "-i", str(gpu)],
                         capture_output=True, text=True).stdout
    return float(out.strip() or 0)


def run_variant(name, bench, gpu, extra, opts, model, out):
    engine = bs.Engine(gpu, extra)
    j = {"id": name, "name": name, "model": model, "max_model_len": opts.pop("max_model_len", 32768),
         "ifbench": {"max_tokens": 4096, "workers": 64, "limit": None},
         "automationbench": {"domains": "simple", "num_examples": None, "max_steps": 50, "max_concurrent": 8}}
    j[bench].update(opts)
    job_dir = out / name
    job_dir.mkdir(parents=True, exist_ok=True)
    try:
        served, how = engine.use(j, job_dir / "service.log")
        before, samples, stop = metrics(engine.port), [], threading.Event()

        def sample():
            while not stop.wait(2):
                m = metrics(engine.port)
                samples.append((m["num_requests_running"], m["num_requests_waiting"], m["kv_cache_usage_perc"], gpu_util(gpu)))

        t = threading.Thread(target=sample, daemon=True)
        t.start()
        mark = time.time()
        scores = bs.RUNNERS[bench](j, job_dir, f"http://127.0.0.1:{engine.port}/v1", job_dir / "bench.log", served)
        wall = time.time() - mark
        stop.set()
        t.join()
        after = metrics(engine.port)
        d = {k: after[k] - before[k] for k in before}
        mean = lambda i: round(sum(s[i] for s in samples) / max(1, len(samples)), 2)
        r = {"variant": name, "bench": bench, "extra": " ".join(extra), **opts, "wall_s": round(wall, 1),
             "engine": how, "gen_tok_s": round(d["generation_tokens_total"] / wall),
             "prompt_tok_s": round(d["prompt_tokens_total"] / wall), "preemptions": int(d["num_preemptions_total"]),
             "prefix_hit": round(d["prefix_cache_hits_total"] / max(1, d["prefix_cache_queries_total"]), 3),
             "running": mean(0), "waiting": mean(1), "kv_usage": mean(2), "gpu_util": mean(3), "scores": scores}
    except Exception as e:
        r = {"variant": name, "bench": bench, "error": str(e)[-500:]}
    finally:
        engine.stop()
    print(json.dumps(r, ensure_ascii=False), flush=True)
    with (out / "results.jsonl").open("a") as f:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(paths.ROOT / "models/Qwen3-4B-Instruct-2507"))
    ap.add_argument("--auto-tasks", type=int, default=32)
    ap.add_argument("--if-prompts", type=int, default=64)
    ap.add_argument("--out", default=str(paths.ROOT / "runs/bench-service/profile" / time.strftime("%Y%m%d-%H%M%S")))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    G = paths.SERVICE_GPUS
    auto = lambda c: {"num_examples": a.auto_tasks, "max_concurrent": c}
    plans = {
        G["automationbench"]: [
            ("auto-bf16-c8", "automationbench", [], auto(8)),
            ("auto-bf16-c16", "automationbench", [], auto(16)),
            ("auto-fp8kv-c16", "automationbench", FP8, auto(16)),
            ("auto-fp8kv-c32", "automationbench", FP8, auto(32)),
            ("auto-fp8kv-c32-mbt16k", "automationbench", FP8 + ["--max-num-batched-tokens", "16384"], auto(32)),
        ],
        G["ifbench"]: [
            ("if-bf16-c16", "ifbench", [], {"limit": a.if_prompts, "workers": 16}),
            ("if-bf16-c64", "ifbench", [], {"limit": a.if_prompts, "workers": 64}),
            ("if-fp8kv-c64", "ifbench", FP8, {"limit": a.if_prompts, "workers": 64}),
        ],
    }
    lease = GpuLease()
    if not lease.try_take(list(plans)):
        raise SystemExit("service GPUs are leased (stop scripts/bench_service.py first)")

    def lane(gpu):
        for name, bench, extra, opts in plans[gpu]:
            run_variant(name, bench, gpu, extra, dict(opts), a.model, out)

    threads = [threading.Thread(target=lane, args=(g,)) for g in plans]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"results: {out / 'results.jsonl'}")


if __name__ == "__main__":
    main()
