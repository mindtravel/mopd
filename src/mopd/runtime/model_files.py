"""Model directories we write (full fine-tuning checkpoints, exported LoRA experts, mergekit merges)
must load exactly like the base model under every environment. One function, finalize(), makes
that so after any writer has put the weights down:

  - config.json, generation config and tokenizer files are the base model's, byte for byte.
    Training runs transformers 5.x, inference (vLLM) 4.55; a 5.x save_pretrained rewrites
    config.json with rope_theta inside a new `rope_parameters` block that 4.55 does not read, so
    it silently fell back to rope_theta=10000 instead of Qwen3's 1e6 and the model produced broken
    syntax and repetition (2026-10-24). 5.x tokenizer configs are likewise unreadable by 4.55.
    Fine-tuning and merging change weights only, so the base's files are always the right ones.
  - the tensor set equals the base's: same names, same shapes. A tied lm_head written as a second
    copy of the embedding (torch.distributed.checkpoint's full state dict does that) is dropped
    after checking it is identical; anything else that differs fails here rather than at eval.
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


def _shards(d):
    """{tensor name: (shard file, shape)} for a safetensors model directory."""
    from safetensors import safe_open
    d = Path(d)
    index = d / "model.safetensors.index.json"
    files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    out = {}
    for f in files:
        with safe_open(str(d / f), "pt") as h:
            for k in h.keys():
                out[k] = (d / f, tuple(h.get_slice(k).get_shape()))
    return out


def _drop_tensor(d, name, shard):
    """Rewrite one shard without `name`; update the index if there is one."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    with safe_open(str(shard), "pt") as h:
        keep = {k: h.get_tensor(k) for k in h.keys() if k != name}
        meta = h.metadata()
    tmp = shard.with_suffix(".partial")
    save_file(keep, str(tmp), metadata=meta)
    tmp.replace(shard)
    index = Path(d) / "model.safetensors.index.json"
    if index.exists():
        idx = json.loads(index.read_text())
        idx["weight_map"].pop(name, None)
        index.write_text(json.dumps(idx, indent=2))


def finalize(model_dir, base):
    """Make model_dir load like `base` everywhere (module docstring). Idempotent. Returns a summary."""
    import torch
    from safetensors import safe_open
    model_dir, base = Path(model_dir), Path(base)
    mine, theirs = _shards(model_dir), _shards(base)
    dropped = None
    extra = set(mine) - set(theirs)
    if extra == {"lm_head.weight"} and json.loads((base / "config.json").read_text()).get("tie_word_embeddings"):
        with safe_open(str(mine["lm_head.weight"][0]), "pt") as h:
            head = h.get_tensor("lm_head.weight")
        with safe_open(str(mine["model.embed_tokens.weight"][0]), "pt") as h:
            embed = h.get_tensor("model.embed_tokens.weight")
        if not torch.equal(head, embed):
            raise ValueError(f"{model_dir}: lm_head.weight differs from the tied embedding")
        _drop_tensor(model_dir, "lm_head.weight", mine["lm_head.weight"][0])
        dropped = "lm_head.weight"; mine.pop("lm_head.weight")
    if set(mine) != set(theirs):
        raise ValueError(f"{model_dir}: tensor names differ from base: {sorted(set(mine) ^ set(theirs))[:5]}")
    bad = {k: (mine[k][1], theirs[k][1]) for k in theirs if mine[k][1] != theirs[k][1]}
    if bad:
        raise ValueError(f"{model_dir}: tensor shapes differ from base: {list(bad.items())[:5]}")
    copy_base_files(base, model_dir)
    return {"tensors": len(mine), "dropped": dropped, "base_files": [n for n in BASE_FILES if (base / n).is_file()]}
