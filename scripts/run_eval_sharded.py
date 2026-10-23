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
import argparse, json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", type=int, nargs="+", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--max-model-len", type=int, default=16384)
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
        gen = [str(ROOT / "mopd_infer/bin/python"), "scripts/generate_benchmark_vllm.py",
               "--model", a.base, "--data", str(data_path), "--out", str(shard_out), "--device", str(gpu),
               "--model-version-id", a.model_version_id, "--protocol-id", a.protocol_id,
               "--max-new-tokens", str(a.max_new_tokens), "--max-model-len", str(a.max_model_len),
               "--student-prompt-format", a.student_prompt_format,
               "--batch-size", "4096", "--max-seqs", "256", "--max-num-batched-tokens", "8192"]
        if a.adapter:
            gen += ["--adapter", a.adapter]
        grade = [PY, "scripts/grade_benchmark_outputs.py", "--data", str(data_path), "--out", str(shard_out)]
        for cmd in (gen, grade):
            if subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode:
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
