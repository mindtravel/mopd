"""Model-directory metadata that must come from the base model, not be re-serialized.

Training runs on transformers 5.x, inference (vLLM) on 4.55. A 5.x save_pretrained rewrites
config.json with rope_theta moved into a new `rope_parameters` block, which 4.55 does not read: it
silently falls back to rope_theta=10000 instead of Qwen3's 1e6, and the model degrades into broken
syntax and repetition (full256 / code-full-v1, 2026-10-24). 5.x tokenizer configs are likewise
unreadable by 4.55. Fine-tuning changes weights only, so every one of these files is copied from
the base byte for byte, and evaluation refuses a model directory whose config differs from it.
"""
import json
import shutil
from pathlib import Path

BASE_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
              "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json")


def copy_base_files(base, out):
    for name in BASE_FILES:
        if (Path(base) / name).is_file():
            shutil.copy2(Path(base) / name, Path(out) / name)


def check_config_matches_base(model_dir, base):
    """Raise unless model_dir's config.json is the base model's (see module docstring)."""
    mine = json.loads((Path(model_dir) / "config.json").read_text())
    theirs = json.loads((Path(base) / "config.json").read_text())
    if mine != theirs:
        diff = sorted(k for k in mine.keys() | theirs.keys() if mine.get(k) != theirs.get(k))
        raise ValueError(f"{model_dir}/config.json differs from the base model's in {diff}: "
                         "re-serialized by another transformers version? copy the base config")
