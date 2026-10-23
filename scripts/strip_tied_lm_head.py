"""Rewrite a full-parameter checkpoint without its tied lm_head, so its tensor set equals the base's.

Checkpoints saved through torch.distributed.checkpoint's full state dict (before fsdp.gather went
per-parameter) list the tied lm_head next to the embedding; mergekit then carries it into the merged
model and finalize_merged_model rejects the extra tensor. The copy is dropped only after checking it
is bit-identical to the embedding, so nothing the model computes changes.

  python scripts/strip_tied_lm_head.py runs/<run>/checkpoint-0256 [...]
"""
import json, sys
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

for d in map(Path, sys.argv[1:]):
    if not json.loads((d / "config.json").read_text()).get("tie_word_embeddings"):
        sys.exit(f"{d}: embeddings are not tied")
    files = sorted(d.glob("*.safetensors"))
    owner = {}
    for f in files:
        with safe_open(str(f), "pt") as h:
            owner.update({k: f for k in h.keys()})
    if "lm_head.weight" not in owner:
        print(f"{d}: no lm_head tensor, nothing to do"); continue
    with safe_open(str(owner["lm_head.weight"]), "pt") as h:
        head = h.get_tensor("lm_head.weight")
    with safe_open(str(owner["model.embed_tokens.weight"]), "pt") as h:
        embed = h.get_tensor("model.embed_tokens.weight")
    if not torch.equal(head, embed):
        sys.exit(f"{d}: lm_head differs from the embedding; not tied in practice, refusing")
    f = owner["lm_head.weight"]
    with safe_open(str(f), "pt") as h:
        keep = {k: h.get_tensor(k) for k in h.keys() if k != "lm_head.weight"}
        meta = h.metadata()
    tmp = f.with_suffix(".partial")
    save_file(keep, str(tmp), metadata=meta)
    tmp.replace(f)
    index = d / "model.safetensors.index.json"
    if index.exists():
        idx = json.loads(index.read_text()); idx["weight_map"].pop("lm_head.weight", None)
        index.write_text(json.dumps(idx, indent=2))
    print(f"{d}: dropped tied lm_head ({tuple(head.shape)}), identical to the embedding")
