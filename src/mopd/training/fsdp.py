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
from pathlib import Path

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from mopd.runtime.model_files import copy_base_files
from mopd.training.batching import TokenLogProbs

def shard(model, reshard_after_forward=True, gather_once_per_step=False):
    """Wrap a causal LM (fp32 parameters on this rank's GPU) for sharded training; returns the
    TokenLogProbs root to train and score with.
    reshard_after_forward=False keeps each layer's gathered bf16 weights from its forward to its
    backward (ZeRO-2-like): one all-gather fewer per micro-batch, +8 GB resident.
    gather_once_per_step additionally keeps them across the step's micro-batches (see step_boundary):
    one all-gather per step however many micro-batches, gradients still reduce-scattered in fp32
    after every micro-batch, so the summed gradient is unchanged. Implies reshard_after_forward=False."""
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    scorer = TokenLogProbs(model)
    units = [fully_shard(layer, mp_policy=policy, reshard_after_forward=reshard_after_forward and not gather_once_per_step)
             for layer in scorer.base.model.layers]
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
    """Collective. Rank 0 writes a plain HF model directory (bf16 safetensors; config and tokenizer
    files copied from the base model) that vLLM and transformers load directly."""
    state = gather(scorer, main_rank)
    if not main_rank:
        return
    path = Path(path)
    scorer.base.save_pretrained(path, state_dict=state, safe_serialization=True)
    copy_base_files(base_model, path)  # config/tokenizer as the inference env reads them (model_files.py)


def write_weights(scorer, path, main_rank):
    """Collective. Rank 0 writes the weights alone as one safetensors file, for a rollout engine to
    load in place (mopd.rollout.vllm_worker). A tied lm_head is left out: it is the embedding."""
    state = gather(scorer, main_rank)
    if not main_rank:
        return
    from safetensors.torch import save_file
    tmp = Path(str(path) + ".partial")
    save_file({k: v.contiguous() for k, v in state.items()}, str(tmp))
    tmp.replace(path)
