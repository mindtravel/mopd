"""Short profiling runs of the bench service's serving path (scripts/bench_service.py): each variant
runs one or both benchmarks (subsets) at once on a service GPU's resident engine and reports wall time with vLLM's own counters (token throughput, preemptions, prefix-cache
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


def run_variant(name, engine, benches, model, out, tasks, prompts):
    """benches: {"ifbench": opts} and/or {"automationbench": opts}, run at once on the engine."""
    j = {"id": name, "name": name, "model": model, "max_model_len": 32768,
         "ifbench": {"max_tokens": 4096, "workers": 64, "limit": prompts},
         "automationbench": {"domains": "simple", "num_examples": tasks, "max_steps": 50, "max_concurrent": 8, "shards": 1}}
    for b, o in benches.items():
        j[b].update(o)
    job_dir = out / name
    job_dir.mkdir(parents=True, exist_ok=True)
    try:
        served, how = engine.use(j, job_dir / "service.log")
        before, samples, stop = metrics(engine.port), [], threading.Event()

        def sample():
            while not stop.wait(2):
                m = metrics(engine.port)
                samples.append((m["num_requests_running"], m["num_requests_waiting"], m["kv_cache_usage_perc"], gpu_util(engine.gpu)))

        t = threading.Thread(target=sample, daemon=True)
        t.start()
        walls, scores, errors = {}, {}, {}

        def one(b):
            mark = time.time()
            try:
                scores[b] = bs.RUNNERS[b](j, job_dir, f"http://127.0.0.1:{engine.port}/v1", job_dir / f"{b}.log", served)
            except Exception as e:
                errors[b] = str(e)[-300:]
            walls[b] = round(time.time() - mark, 1)

        mark = time.time()
        threads = [threading.Thread(target=one, args=(b,)) for b in benches]
        for x in threads:
            x.start()
        for x in threads:
            x.join()
        wall = time.time() - mark
        stop.set()
        t.join()
        d = {k: v - before[k] for k, v in metrics(engine.port).items()}
        mean = lambda i: round(sum(x[i] for x in samples) / max(1, len(samples)), 2)
        r = {"variant": name, "gpu": engine.gpu, "benches": benches, "wall_s": round(wall, 1), "per_bench_s": walls,
             "engine": how, "gen_tok_s": round(d["generation_tokens_total"] / wall),
             "prompt_tok_s": round(d["prompt_tokens_total"] / wall), "preemptions": int(d["num_preemptions_total"]),
             "prefix_hit": round(d["prefix_cache_hits_total"] / max(1, d["prefix_cache_queries_total"]), 3),
             "running": mean(0), "waiting": mean(1), "kv_usage": mean(2), "gpu_util": mean(3),
             "scores": scores, **({"errors": errors} if errors else {})}
    except Exception as e:
        engine.stop()
        r = {"variant": name, "error": str(e)[-500:]}
    print(json.dumps(r, ensure_ascii=False), flush=True)
    with (out / "results.jsonl").open("a") as f:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(paths.ROOT / "models/Qwen3-4B-Instruct-2507"))
    ap.add_argument("--auto-tasks", type=int, default=64)
    ap.add_argument("--if-prompts", type=int, default=64)
    ap.add_argument("--out", default=str(paths.ROOT / "runs/bench-service/profile" / time.strftime("%Y%m%d-%H%M%S")))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    g6, g7 = paths.SERVICE_GPUS
    IF, A1, A4 = {}, {"shards": 1}, {"shards": 4}
    rounds = [  # each round: {gpu: (variant, benches)}, the GPUs in parallel
        {g6: ("ifbench-alone", {"ifbench": IF}), g7: ("auto-alone-1proc", {"automationbench": A1})},
        {g6: ("merged-auto-1proc", {"ifbench": IF, "automationbench": A1}),
         g7: ("merged-auto-4proc", {"ifbench": IF, "automationbench": A4})},
        {g7: ("auto-alone-4proc", {"automationbench": A4})},
    ]
    lease = GpuLease()
    if not lease.try_take([g6, g7]):
        raise SystemExit("service GPUs are leased (stop scripts/bench_service.py first)")
    engines = {g: bs.Engine(g) for g in (g6, g7)}
    try:
        for rnd in rounds:
            threads = [threading.Thread(target=run_variant, args=(name, engines[g], benches, a.model, out, a.auto_tasks, a.if_prompts))
                       for g, (name, benches) in rnd.items()]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
    finally:
        for e in engines.values():
            e.stop()
    print(f"results: {out / 'results.jsonl'}")


if __name__ == "__main__":
    main()
