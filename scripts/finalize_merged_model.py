"""Make a model directory written by another tool (mergekit) load exactly like its base:
mopd.runtime.model_files.finalize -- base config/tokenizer files, tensor set checked against the base.

  python scripts/finalize_merged_model.py --base models/Qwen3-4B-Base models/merged/<name>
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mopd.runtime.model_files import finalize


def main():
    p = argparse.ArgumentParser(); p.add_argument("--base", required=True); p.add_argument("model"); a = p.parse_args()
    print(json.dumps({"model": a.model, **finalize(a.model, a.base)}))


if __name__ == "__main__":
    main()
