"""Full-parameter training of an HF causal LM across GPUs with FSDP2 (student_backend "hf_full").

4B parameters in fp32 plus fp32 gradients and AdamW state is ~64 GB: no single 32 GB GPU holds it,
so parameters, gradients and optimizer state are sharded over the ranks. Compute is bf16
(MixedPrecisionPolicy); master weights, gradient reduction and the optimizer stay fp32.

Units: one per decoder layer, plus the root (mopd.training.batching.TokenLogProbs), which keeps the
embedding, final norm and lm_head -- the embedding and lm_head are one tied tensor in Qwen3, so they
must live in the same unit. Gradients are reduce-scattered as a SUM (divide factor 1): train_sft's
per-sample scales are computed over the whole global step, so summing per-rank gradients is what
reproduces the single-GPU gradient, the same contract as its LoRA all-reduce path.
"""

import json
from pathlib import Path

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from mopd.runtime.model_files import finalize
from mopd.training.batching import TokenLogProbs


def shard(model, reshard_after_forward=True, gather_once_per_step=False, reduce_dtype="float32"):
    """Wrap a causal LM (fp32 parameters on this rank's GPU) for sharded training; returns the
    TokenLogProbs root to train and score with.
    reshard_after_forward=False keeps each layer's gathered bf16 weights from its forward to its
    backward (ZeRO-2-like): one all-gather fewer per micro-batch, +8 GB resident.
    gather_once_per_step additionally keeps them across the step's micro-batches (see step_boundary):
    one all-gather per step however many micro-batches, gradients still reduce-scattered in fp32
    after every micro-batch, so the summed gradient is unchanged. Implies reshard_after_forward=False."""
    # reduce_dtype "bfloat16" halves the gradient reduce-scatter bytes (sums of a few ranks' bf16
    # gradients; the optimizer and master weights stay fp32)
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=getattr(torch, reduce_dtype))
    scorer = TokenLogProbs(model)
    units = [
        fully_shard(layer, mp_policy=policy, reshard_after_forward=reshard_after_forward and not gather_once_per_step)
        for layer in scorer.base.model.layers
    ]
    units.append(fully_shard(scorer, mp_policy=policy))
    for unit in units:
        unit.set_reduce_scatter_divide_factor(1.0)
    scorer.gather_once_per_step = gather_once_per_step
    return scorer


def before_micro_batch(scorer, last):
    """gather_once_per_step: keep parameters unsharded after every backward but the step's last, so
    the next micro-batch's forward needs no all-gather; the last backward reshards them for the
    optimizer step. No-op otherwise."""
    if getattr(scorer, "gather_once_per_step", False):
        scorer.set_reshard_after_backward(last)


def local(tensor):
    """This rank's shard of a sharded (DTensor) parameter or gradient; plain tensors as they are."""
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def gather(scorer, main_rank):
    """Collective: every rank must call. The whole model's weights as bf16 CPU tensors under HF
    names on rank 0 (None elsewhere). One parameter at a time, cast to bf16 on the GPU *before* the
    all-gather: half the bytes of gathering the fp32 master copy and converting on the CPU (that
    took ~21 s per MOPD step, about as long as the rollouts), and never more than one full tensor
    (the 0.8 GB embedding at most) on any GPU. A tied lm_head is the embedding and appears once."""
    state = {}
    for name, p in scorer.named_parameters():
        full = p.detach().to(torch.bfloat16)
        full = full.full_tensor() if hasattr(full, "full_tensor") else full
        if main_rank:
            state[name.removeprefix("base.")] = full.cpu()
        del full
    return state if main_rank else None


def save(scorer, path, base_model, main_rank):
    """Collective. Rank 0 writes a plain HF model directory (bf16 safetensors, finalized against the
    base model) that vLLM and transformers load directly."""
    state = gather(scorer, main_rank)
    if not main_rank:
        return
    path = Path(path)
    scorer.base.save_pretrained(path, state_dict=state, safe_serialization=True)
    finalize(path, base_model)  # base config/tokenizer, tensor set checked against the base (model_files.py)


class SharedWeights:
    """The student's bf16 weights in one flat buffer in shared memory (a /dev/shm file), page-locked,
    rewritten in place every step for rollout engines to map (mopd.rollout.engine_worker).

    Measured on this server while another job saturated host memory bandwidth: gathering the 8 GB
    of bf16 weights to rank 0's GPU takes 0.5 s, but copying them to ordinary (pageable) host memory
    took 10-18 s and serializing them with safetensors another 17-30 s -- every CPU memcpy is slow
    when the host is busy. Here each gathered tensor is DMA-copied straight from the GPU into the
    registered shared buffer: no pageable copy, no serialization, one buffer for the whole run.
    Layout: <path> holds the tensors back to back; <path>.json their names, shapes and offsets."""

    def __init__(self, scorer, path, main_rank):
        self.path, self.main = Path(path), main_rank
        self.layout, offset = [], 0
        for name, p in scorer.named_parameters():
            n = p.numel()
            self.layout.append(
                {"name": name.removeprefix("base."), "shape": list(p.shape), "offset": offset, "numel": n}
            )
            offset += n
        self.numel, self.buffer = offset, None
        if main_rank:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.buffer = torch.from_file(str(self.path), shared=True, size=offset, dtype=torch.bfloat16)
            rc = torch.cuda.cudart().cudaHostRegister(self.buffer.data_ptr(), offset * 2, 0)
            if int(rc) != 0:
                raise RuntimeError(f"cudaHostRegister failed ({rc}) for {self.path}")
            Path(str(self.path) + ".json").write_text(json.dumps({"dtype": "bfloat16", "tensors": self.layout}))

    def write(self, scorer):
        """Collective: every rank must call. Rank 0's buffer holds the current weights on return."""
        for (name, p), item in zip(scorer.named_parameters(), self.layout):
            full = p.detach().to(torch.bfloat16)
            full = full.full_tensor() if hasattr(full, "full_tensor") else full
            if self.main:
                self.buffer[item["offset"] : item["offset"] + item["numel"]].view(item["shape"]).copy_(
                    full, non_blocking=True
                )
            del full
        if self.main:
            torch.cuda.synchronize()

    def close(self):
        if self.buffer is not None:
            torch.cuda.cudart().cudaHostUnregister(self.buffer.data_ptr())
            self.buffer = None


def read_shared_weights(path):
    """(name, tensor) views into a SharedWeights buffer, mapped without copying (rollout side)."""
    meta = json.loads(Path(str(path) + ".json").read_text())
    total = sum(t["numel"] for t in meta["tensors"])
    flat = torch.from_file(str(path), shared=True, size=total, dtype=torch.bfloat16)
    for t in meta["tensors"]:
        yield t["name"], flat[t["offset"] : t["offset"] + t["numel"]].view(t["shape"])
