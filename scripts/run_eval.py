"""Generate + grade one raw model (base, optionally with a LoRA adapter) on one benchmark, on one
or more GPUs: rows go to whichever GPU has room (scripts/generate_benchmark_vllm.py). For ad-hoc
evals outside a sweep, e.g. re-scoring a checkpoint with sampling or a bigger generation budget.

  python scripts/run_eval.py --base models/Qwen3-4B-Base --adapter runs/.../checkpoint-0128 \
    --data data/full_eval/lcb-dev-v1/lcbdev.jsonl --out results/.../probe --gpus 0 1 2 3 4 5 6 7 \
    --max-new-tokens 16384 --temperature 0.6 --top-p 0.95 \
    --model-version-id rstar-seed-concise:step-0128 --protocol-id lcbdev-plainqa-t0.6p0.95-16384
"""
import argparse, os, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mopd.evaluation.eval_commands import eval_commands


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
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    a = ap.parse_args()
    for cmd, extra in eval_commands(data=a.data, out=a.out, device=a.gpus, protocol_id=a.protocol_id,
                                    max_new_tokens=a.max_new_tokens, max_model_len=a.max_model_len,
                                    student_prompt_format=a.student_prompt_format, model=a.base,
                                    model_version_id=a.model_version_id, adapter=a.adapter,
                                    temperature=a.temperature, top_p=a.top_p):
        subprocess.run(cmd, cwd=ROOT, env={**os.environ, **extra}, check=True)
    print((Path(a.out) / "summary.json").read_text())


if __name__ == "__main__":
    main()
