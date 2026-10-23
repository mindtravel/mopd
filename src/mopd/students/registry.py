"""Student construction and checkpointing by backend name.

Adding a student backend means adding one loader and one saver here; the training
loop, rollout layer and evaluation never branch on backend identity.
"""
from pathlib import Path

import torch

BACKENDS = ("hf_lora", "hf_full", "fairy2i_qat")
# Backends trained sharded across ranks (mopd.training.fsdp): FSDP reduces their gradients, and
# their checkpoint save is a collective (every rank calls save_student; rank 0 writes).
SHARDED = {"hf_full"}


def is_sharded(config):
    return config.get("student_backend", "hf_lora") in SHARDED


def _load_hf_lora(config):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mopd.backends.student_adapter import attach_student_adapter

    tokenizer = AutoTokenizer.from_pretrained(config["student"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        config["student"], dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    ).to(config["student_device"])
    return tokenizer, attach_student_adapter(model, config)


def _load_hf_full(config):
    """Every weight trainable, fp32 master copy; sharded across ranks by mopd.training.fsdp.shard."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["student"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        config["student"], dtype=torch.float32, attn_implementation="sdpa", local_files_only=True
    ).to(config["student_device"])
    return tokenizer, model


def _load_fairy2i_qat(config):
    from mopd.backends.fairy2i_qat_adapter import load_fairy2i_qat_student

    return load_fairy2i_qat_student(
        config["student"], config["student_device"],
        trainable_scales=True, update_path=config.get("student_adapter"),
    )


LOADERS = {"hf_lora": _load_hf_lora, "hf_full": _load_hf_full, "fairy2i_qat": _load_fairy2i_qat}


def build_student(config):
    backend = config.get("student_backend", "hf_lora")
    if backend not in LOADERS:
        raise ValueError(f"Unknown student_backend {backend!r}; expected one of {BACKENDS}")
    tokenizer, model = LOADERS[backend](config)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if config.get("gradient_checkpointing", True):  # off trades memory for no recompute (faster)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return tokenizer, model


def save_student(model, config, path, step, main_rank=True):
    """Persist one checkpoint. LoRA writes adapters; QAT writes quantized scales; full fine-tuning
    writes a whole HF model directory (collective, see SHARDED)."""
    path = Path(path)
    if config.get("student_backend") == "hf_full":
        from mopd.training import fsdp
        fsdp.save(model, path, config["student"], main_rank)
    elif config.get("student_backend") == "fairy2i_qat":
        from mopd.backends.fairy2i_qat_adapter import save_fairy2i_qat_update
        save_fairy2i_qat_update(model, path, base_model=config["student"], step=step)
    else:
        model.save_pretrained(path, safe_serialization=True)
    return path
