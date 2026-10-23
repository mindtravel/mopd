"""Evaluate one model on one benchmark across several GPUs in parallel, generous generation budget.

Splits the benchmark file into N interleaved shards (round-robin by row index, not contiguous
chunks: generation length varies a lot row to row, e.g. reasoning-chain code models, and a
contiguous split would pile the slow rows onto one GPU), runs generate_benchmark_vllm.py +
grade_benchmark_outputs.py per shard on its own GPU, then merges the graded shards with
merge_benchmark_results.py into one summary. Use this instead of evaluate_model_version.py's
single-GPU path when a model's natural outputs are too long for the usual 4096/8192-token budget
to be fair (high truncation = the score reflects the cap more than the model).

  python scripts/run_eval_sharded.py --adapter runs/.../checkpoint-0256 --base models/Qwen3-4B-Base \
    --data data/full_eval/lcb-dev-v1/lcbdev.jsonl --out results/.../checkpoint-0256-sharded \
    --gpus 0 2 3 4 5 6 7 --max-new-tokens 24576 --max-model-len 32768 \
    --model-version-id ocr-trace:step-0256 --protocol-id lcbdev-plainqa-greedy24576
"""
import argparse, json, os, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mopd.evaluation.eval_commands import eval_commands, PY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", type=int, nargs="+", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--max-model-len", type=int, help="default: max(16384, max-new-tokens+4096)")
    ap.add_argument("--student-prompt-format", default="plain_qa_v1")
    ap.add_argument("--model-version-id", required=True)
    ap.add_argument("--protocol-id", required=True)
    a = ap.parse_args()

    rows = [json.loads(l) for l in Path(a.data).read_text().splitlines() if l.strip()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    shard_dir = out / "shards"
    shard_dir.mkdir()
    n = len(a.gpus)
    shard_paths = []
    for i, gpu in enumerate(a.gpus):
        shard = rows[i::n]  # interleaved, not contiguous: spreads long/short rows evenly
        if not shard:
            continue
        path = shard_dir / f"data-{i}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in shard))
        shard_paths.append((gpu, path, shard_dir / f"out-{i}"))

    print(f"{len(rows)} rows across {len(shard_paths)} GPUs: {[len(rows[i::n]) for i in range(len(shard_paths))]}", flush=True)

    def run(gpu, data_path, shard_out):
        shard_out.mkdir(parents=True, exist_ok=True)
        log = (shard_out.parent / f"{shard_out.name}.log").open("a")
        cmds = eval_commands(data=data_path, protocol_id=a.protocol_id, max_new_tokens=a.max_new_tokens, device=gpu,
            out=shard_out, student_prompt_format=a.student_prompt_format, model=a.base, model_version_id=a.model_version_id,
            adapter=a.adapter, max_model_len=a.max_model_len)
        for cmd, extra in cmds:
            if subprocess.run(cmd, cwd=ROOT, env={**os.environ, **extra}, stdout=log, stderr=subprocess.STDOUT).returncode:
                return False
        return True

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(len(shard_paths)) as pool:
        ok = list(pool.map(lambda t: run(*t), shard_paths))
    if not all(ok):
        raise SystemExit(f"shard(s) failed: {[p for (g, d, p), k in zip(shard_paths, ok) if not k]}")

    merged = out / "merged"
    subprocess.run([PY, "scripts/merge_benchmark_results.py", "--data", a.data, "--out", str(merged),
                    *[str(p) for _, _, p in shard_paths]], cwd=ROOT, check=True)
    print((merged / "summary.json").read_text())


if __name__ == "__main__":
    main()
