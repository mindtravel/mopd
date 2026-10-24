"""Publish the student's current weights for rollout workers, independent of how it is trained.

A policy descriptor {"kind", "path", "id", "version"} tells mopd.rollout.vllm_worker what to run:
  lora: an adapter directory, loaded with add_lora on top of the base model (cheap, ~0.2 s/step);
  full: one page-locked bf16 buffer in /dev/shm (fsdp.SharedWeights), rewritten in place every step
        and mapped by the rollout engines, which load it with model.load_weights.
`id` is what workers echo back so a reply can be matched to the exact weights it used: the adapter
sha256 for LoRA; for full weights the version (hashing 8 GB every step would cost more than the
load; the trainer only rewrites the buffer after every rollout of the previous version returned).
"""

import hashlib
import os
from pathlib import Path

from mopd.students.registry import is_sharded

_shared = None  # fsdp.SharedWeights of this process's run (full-parameter students)


def shm_dir(config):
    return Path("/dev/shm/mopd") / Path(config["output"]).name


def cleanup(config):
    """Free the RAM held by a run's published full-weight buffer (call once training ends)."""
    import shutil

    global _shared
    if _shared is not None:
        _shared.close()
        _shared = None

    shutil.rmtree(shm_dir(config), ignore_errors=True)


def publish(student, config, root, version, tokenizer, main_rank):
    """Collective for sharded students (every rank calls); returns the descriptor on rank 0."""
    if is_sharded(config):
        from mopd.training import fsdp

        global _shared
        if _shared is None:  # one page-locked shared buffer per run, rewritten every step
            _shared = fsdp.SharedWeights(student, shm_dir(config) / "policy.bf16", main_rank)
        _shared.write(student)
        if not main_rank:
            return None
        return {"kind": "full", "path": str(_shared.path), "id": f"full:{version}", "version": version}
    adapter = Path(root) / f"adapter-{version:06d}"
    if adapter.exists():
        raise ValueError("Adapter versions must be immutable and unique")
    student.save_pretrained(adapter, safe_serialization=True)
    tokenizer.save_pretrained(adapter)
    digest = hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
    return {"kind": "lora", "path": str(adapter), "id": digest, "version": version}
