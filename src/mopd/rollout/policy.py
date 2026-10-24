"""Publish the student's current weights for rollout workers, independent of how it is trained.

A policy descriptor {"kind", "path", "id", "version"} tells mopd.rollout.engine_worker what to run:
  lora: an adapter directory, loaded with add_lora on top of the base model (cheap, ~0.2 s/step);
  full: one page-locked bf16 buffer in /dev/shm (fsdp.SharedWeights), rewritten in place every step
        and mapped by the rollout engines, which load it with model.load_weights.
`id` is what workers echo back so a reply can be matched to the exact weights it used: the adapter
sha256 for LoRA; for full weights the version (hashing 8 GB every step would cost more than the
load; the trainer only rewrites the buffer after every rollout of the previous version returned).
"""

from mopd import paths
import hashlib
from pathlib import Path

from mopd.students.registry import is_sharded


def shm_dir(config):
    return paths.SHM / Path(config["output"]).name


class Publisher:
    """Publishes each policy version of one run's student for its rollout workers. Owns the run's
    page-locked shared buffer for full-parameter students (created on first use, freed by close())."""

    def __init__(self, config, root, tokenizer, main_rank):
        self.c, self.root, self.tok, self.main = config, Path(root), tokenizer, main_rank
        self.shared = None

    def publish(self, student, version):
        """Collective for sharded students (every rank calls); returns the descriptor on rank 0."""
        if is_sharded(self.c):
            from mopd.training import fsdp

            if self.shared is None:
                self.shared = fsdp.SharedWeights(student, shm_dir(self.c) / "policy.bf16", self.main)
            self.shared.write(student)
            if not self.main:
                return None
            return {"kind": "full", "path": str(self.shared.path), "id": f"full:{version}", "version": version}
        if not self.main:
            return None
        adapter = self.root / f"adapter-{version:06d}"
        if adapter.exists():
            raise ValueError("Adapter versions must be immutable and unique")
        student.save_pretrained(adapter, safe_serialization=True)
        self.tok.save_pretrained(adapter)
        digest = hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
        return {"kind": "lora", "path": str(adapter), "id": digest, "version": version}

    def close(self):
        """Free the shared buffer's page lock and RAM (call once training ends)."""
        import shutil

        if self.shared is not None:
            self.shared.close()
            self.shared = None
        if self.main:
            shutil.rmtree(shm_dir(self.c), ignore_errors=True)
