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
import shutil
from pathlib import Path

import torch
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from mopd.training.batching import TokenLogProbs

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                   "special_tokens_map.json", "added_tokens.json", "generation_config.json")


def shard(model, reshard_after_forward=True):
    """Wrap a causal LM (fp32 parameters on this rank's GPU) for sharded training; returns the
    TokenLogProbs root to train and score with. reshard_after_forward=False keeps each layer's
    gathered bf16 weights until its backward (ZeRO-2-like): more memory, one all-gather fewer."""
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    scorer = TokenLogProbs(model)
    units = [fully_shard(layer, mp_policy=policy, reshard_after_forward=reshard_after_forward)
             for layer in scorer.base.model.layers]
    units.append(fully_shard(scorer, mp_policy=policy))
    for unit in units:
        unit.set_reduce_scatter_divide_factor(1.0)
    return scorer


def local(tensor):
    """This rank's shard of a sharded (DTensor) parameter or gradient; plain tensors as they are."""
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def gather(scorer, main_rank):
    """Collective: every rank must call. The whole model's weights as bf16 CPU tensors under HF
    names on rank 0 (None elsewhere)."""
    state = get_model_state_dict(scorer, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if not main_rank:
        return None
    return {k.removeprefix("base."): v.to(torch.bfloat16) for k, v in state.items()}


def save(scorer, path, base_model, main_rank):
    """Collective. Rank 0 writes a plain HF model directory (bf16 safetensors, config and tokenizer
    files copied from the base model) that vLLM and transformers load directly."""
    state = gather(scorer, main_rank)
    if not main_rank:
        return
    path = Path(path)
    scorer.base.save_pretrained(path, state_dict=state, safe_serialization=True)
    for name in TOKENIZER_FILES:
        if (Path(base_model) / name).is_file():
            shutil.copy2(Path(base_model) / name, path / name)


def write_weights(scorer, path, main_rank):
    """Collective. Rank 0 writes the weights alone as one safetensors file, for a rollout engine to
    load in place (mopd.rollout.vllm_worker). A tied lm_head is left out: it is the embedding."""
    state = gather(scorer, main_rank)
    if not main_rank:
        return
    from safetensors.torch import save_file
    if scorer.base.config.tie_word_embeddings:
        state.pop("lm_head.weight", None)
    tmp = Path(str(path) + ".partial")
    save_file({k: v.contiguous() for k, v in state.items()}, str(tmp))
    tmp.replace(path)
