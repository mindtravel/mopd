"""Evaluate one model on one benchmark across any number of GPUs: the one eval path for
run_sweep.py and run_eval_sharded.py.

One GPU: the plain generate+grade commands from eval_commands(), writing straight into `out`.
Several: the benchmark is split into interleaved shards (rows[i::n], not contiguous chunks -- long
reasoning-chain rows cluster, and a contiguous split piles them onto one GPU), each shard is
generated and graded on its own GPU under out/shards/<i>/, and the graded shards are merged by
scripts/merge_benchmark_results.py into out/{cases.jsonl, summary.json} -- the same files a
single-GPU eval leaves, so callers never care how many GPUs ran it. summary.json is written last
and is the completion marker.

Resume: each shard resumes its own generated.jsonl. A shard's data file is pinned by sha256 in its
generation-config.json, so a different GPU count means different shard files; out/shards is then
discarded and regenerated rather than mis-resumed.
"""
import json, os, shutil, subprocess, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mopd.evaluation.eval_commands import eval_commands, PY, ROOT
from mopd.runtime import progress


def _run(cmds, log):
    return all(subprocess.run(cmd, cwd=ROOT, env={**os.environ, **extra}, stdout=log, stderr=subprocess.STDOUT).returncode == 0
               for cmd, extra in cmds)


def evaluate(*, data, out, gpus, log=None, **command_args):
    """command_args are eval_commands()'s model/protocol arguments (protocol_id, max_new_tokens,
    model or model_version+checkpoint, ...). Returns True on success."""
    out, gpus = Path(out), list(gpus)
    if (out / "summary.json").exists():
        return True
    out.mkdir(parents=True, exist_ok=True)
    if len(gpus) == 1:
        return _run(eval_commands(data=data, out=out, device=gpus[0], **command_args), log)

    rows = [l for l in Path(data).read_text().splitlines() if l.strip()]
    n = min(len(gpus), len(rows))
    shards = out / "shards"
    layout = shards / "layout.json"
    if shards.exists() and (not layout.exists() or json.loads(layout.read_text())["shards"] != n):
        shutil.rmtree(shards)
    shards.mkdir(exist_ok=True)
    layout.write_text(json.dumps({"shards": n, "rows": len(rows)}))
    jobs = []
    for i, gpu in enumerate(gpus[:n]):
        d = shards / str(i)
        d.mkdir(exist_ok=True)
        (shards / f"{i}.jsonl").write_text("".join(l + "\n" for l in rows[i::n]))
        jobs.append((gpu, shards / f"{i}.jsonl", d))

    stop = threading.Event()

    def write_progress():  # one total for the whole benchmark, alongside each shard's own progress file
        done = sum(sum(1 for _ in f.open()) for f in (d / "generated.jsonl" for _, _, d in jobs) if f.exists())
        progress.write(out / "generation-progress.json", done=done, total=len(rows), shards=n)

    def report():
        while not stop.wait(15):
            write_progress()

    def shard(job):
        gpu, shard_data, d = job
        with (shards / f"{d.name}.log").open("a") as shard_log:
            return _run(eval_commands(data=shard_data, out=d, device=gpu, **command_args), shard_log)

    monitor = threading.Thread(target=report, daemon=True)
    monitor.start()
    try:
        with ThreadPoolExecutor(n) as pool:
            ok = list(pool.map(shard, jobs))
    finally:
        stop.set()
        write_progress()
    if not all(ok):
        if log:
            log.write(f"sharded eval: failed shards {[d.name for (_, _, d), k in zip(jobs, ok) if not k]}, logs in {shards}\n")
        return False

    merged = out / "merged.partial"
    shutil.rmtree(merged, ignore_errors=True)
    if subprocess.run([PY, "scripts/merge_benchmark_results.py", "--data", str(data), "--out", str(merged),
                       *[str(d) for _, _, d in jobs]], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode:
        return False
    (merged / "cases.jsonl").replace(out / "cases.jsonl")
    (merged / "summary.json").replace(out / "summary.json")
    merged.rmdir()
    return True
