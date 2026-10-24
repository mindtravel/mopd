"""On-policy rollouts on resident vLLM engines (mopd.rollout.engine_pool): all of a step's
generation finishes on the published policy version before the student updates.

stream() yields trajectories in finish order (teacher scoring can start early); generate() returns the
whole step. Both take a published policy descriptor (mopd.rollout.policy.Publisher), not the student:
publishing is the trainer's job, and a collective for sharded students. Rows go to whichever engine
has room, capped at a fair share, so the long tail of a step spreads over every rollout GPU.
"""

import time
from pathlib import Path

from transformers import AutoTokenizer

from mopd import paths
from mopd.data.domains import student_prompt
from mopd.evaluation.context import default_max_model_len
from mopd.rollout.base import RolloutStream
from mopd.rollout.engine_pool import EnginePool
from mopd.rollout.vllm_protocol import validate_item
from mopd.students.registry import is_sharded


def lora_rank(c):
    """vLLM preallocates LoRA buffers at max_lora_rank and accepts only these sizes."""
    r = c.get("lora", {}).get("rank", 16)
    return next(size for size in (8, 16, 32, 64, 128, 256, 320, 512) if size >= r)


class VllmRollouts:
    def __init__(self, c, student=None):
        if not c.get("rollout_devices"):
            raise ValueError("No rollout GPUs configured")
        self.c = c
        root = Path(c["output"]) / "vllm_runtime"
        self.tokenizer = AutoTokenizer.from_pretrained(c["student"], local_files_only=True)
        engine = {
            "model": c["student"],
            "max_model_len": c.get("rollout_max_model_len") or default_max_model_len(c["max_new_tokens"]),
            "max_seqs": c.get("rollout_max_seqs", 256),
            "max_num_batched_tokens": c.get("rollout_max_num_batched_tokens", 8192),
            "memory": c.get("vllm_memory_utilization", 0.85),
            "prefix_caching": False,
            "max_lora_rank": None if is_sharded(c) else lora_rank(c),
        }
        self.pool = EnginePool(
            c["rollout_devices"], engine, root, root / "compile_cache", python=c.get("vllm_python", str(paths.INFER_PY))
        )
        if set(self.pool.eos.values()) != {self.tokenizer.eos_token_id}:
            self.pool.close()
            raise RuntimeError("Worker/tokenizer EOS mismatch")

    def close(self):
        self.pool.close()

    def stream(self, policy, rows, version, *, sample=True, min_new_tokens=0):
        if policy["version"] != version:
            raise ValueError("published policy is not this step's version")
        inputs = [
            {
                "index": i,
                "prefix": self.tokenizer.encode(student_prompt(self.tokenizer, r, self.c["student_prompt_format"])),
                "row": r,
            }
            for i, r in enumerate(rows)
        ]
        session = {
            "max_tokens": self.c["max_new_tokens"],
            "temperature": 1.0 if sample else 0.0,
            "top_p": 1.0,
            "seed": self.c["seed"] + version,
            "min_tokens": min_new_tokens,
            "logprobs": self.c.get("debug_logprobs"),
        }
        eos = self.tokenizer.eos_token_id

        def source():
            for key, m, device in self.pool.run([(x["index"], x["prefix"]) for x in inputs], session, policy):
                item = {"answer": m["token_ids"], "first_logprobs": m.get("first_logprobs")}
                yield {
                    **validate_item(item, inputs[key], version, self.c["max_new_tokens"], eos),
                    "rollout_device": f"cuda:{device}",
                }
            if set(self.pool.stats["policy_id"].values()) != {policy["id"]}:
                raise ValueError(f"engines ran {self.pool.stats['policy_id']}, expected {policy['id']}")

        return RolloutStream(source(), lambda: {"backend": "vllm-pool", **self.pool.stats})

    def generate(self, policy, rows, version, *, sample=True, min_new_tokens=0):
        started = time.perf_counter()
        stream = self.stream(policy, rows, version, sample=sample, min_new_tokens=min_new_tokens)
        output = sorted(stream, key=lambda r: r["index"])
        return output, time.perf_counter() - started, stream.stats
