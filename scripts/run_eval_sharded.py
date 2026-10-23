"""Evaluate one model on one benchmark across several GPUs (mopd.evaluation.sharded): command-line
wrapper for ad-hoc evals outside a sweep, e.g. re-scoring a checkpoint with a bigger generation
budget when a high truncation count means the score reflects the cap more than the model.

  python scripts/run_eval_sharded.py --adapter runs/.../checkpoint-0256 --base models/Qwen3-4B-Base \
    --data data/full_eval/lcb-dev-v1/lcbdev.jsonl --out results/.../checkpoint-0256-sharded \
    --gpus 0 2 3 4 5 6 7 --max-new-tokens 24576 --max-model-len 32768 \
    --model-version-id ocr-trace:step-0256 --protocol-id lcbdev-plainqa-greedy24576
"""
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mopd.evaluation import sharded


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
    ok = sharded.evaluate(data=a.data, out=a.out, gpus=a.gpus, log=sys.stdout, protocol_id=a.protocol_id,
                          max_new_tokens=a.max_new_tokens, max_model_len=a.max_model_len,
                          student_prompt_format=a.student_prompt_format, model=a.base,
                          model_version_id=a.model_version_id, adapter=a.adapter)
    if not ok:
        raise SystemExit("sharded eval failed")
    print((Path(a.out) / "summary.json").read_text())


if __name__ == "__main__":
    main()
