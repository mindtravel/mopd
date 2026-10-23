"""Publish the student's current weights for rollout workers, independent of how it is trained.

A policy descriptor {"kind", "path", "id", "version"} tells mopd.rollout.vllm_worker what to run:
  lora: an adapter directory, loaded with add_lora on top of the base model (cheap, ~0.2 s/step);
  full: a safetensors file of every weight, loaded in place with model.load_weights. Written to
        /dev/shm (RAM) because it is rewritten every step; only the newest version is kept.
`id` is what workers echo back so a reply can be matched to the exact weights it used: the adapter
sha256 for LoRA; for full weights version and byte size (hashing 8 GB every step would cost more
than the load itself, and each version gets a fresh file name).
"""
import hashlib
import os
from pathlib import Path

from mopd.students.registry import is_sharded


def publish(student, config, root, version, tokenizer, main_rank):
    """Collective for sharded students (every rank calls); returns the descriptor on rank 0."""
    if is_sharded(config):
        from mopd.training import fsdp
        shm = Path("/dev/shm/mopd") / Path(config["output"]).name
        path = shm / f"policy-{version:06d}.safetensors"
        if main_rank:
            shm.mkdir(parents=True, exist_ok=True)
        fsdp.write_weights(student, path, main_rank)
        if not main_rank:
            return None
        for old in shm.glob("policy-*.safetensors"):
            if old != path:
                old.unlink()
        return {"kind": "full", "path": str(path), "id": f"full:{version}:{os.path.getsize(path)}", "version": version}
    adapter = Path(root) / f"adapter-{version:06d}"
    if adapter.exists():
        raise ValueError("Adapter versions must be immutable and unique")
    student.save_pretrained(adapter, safe_serialization=True)
    tokenizer.save_pretrained(adapter)
    digest = hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
    return {"kind": "lora", "path": str(adapter), "id": digest, "version": version}
