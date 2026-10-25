"""The controlled training fields of a comparison, shared by experiment specs
(scripts/make_experiment.py) and sweep specs (scripts/run_sweep.py): one list of known control keys
(a typo is an error, not a silently ignored setting) and one function that turns controls into a
training config's common fields, in a fixed order (generated configs are committed and checked
byte for byte)."""

CONTROL_KEYS = {
    # every comparison
    "student", "student_backend", "student_prompt_format", "seed", "steps", "samples_per_step", "lr",
    "lr_schedule", "save_every", "lr_reason",
    # training system (same result, different speed/memory): mopd.training.fsdp, students.registry
    "fsdp", "gradient_checkpointing", "lora", "micro_batch_tokens",
    # experiment specs
    "micro_batch_size", "sft_max_seq_len", "mopd_max_new_tokens", "mopd_generation_batch_size",
    "mopd_vllm_memory_utilization", "teacher_prompt_format", "mopd_rollout_async",
}
SYSTEM_KEYS = ("fsdp", "gradient_checkpointing", "lora", "micro_batch_tokens")

# Default lr per (student backend, training method): a LoRA update moves the model far less per unit
# lr than a full-parameter one, and the MOPD reverse-KL loss has another gradient scale than SFT
# cross-entropy, so lr is never carried across those boundaries. Evidence and scope:
# docs/training/learning-rate.md. None: no default yet -- set lr with an lr_reason.
LR_DEFAULTS = {
    ("hf_full", "sft"): 2e-6,  # code-full-v1: lcb-dev-v3 ck128 30.3 vs 25.8 at 1e-5 (32 samples/step, 256 steps)
    ("hf_lora", "sft"): 5e-5,  # scale256 / code-data-v1 (not swept)
    ("hf_lora", "mopd"): 5e-5,  # scale256 MOPD arms (not swept)
    ("hf_full", "mopd"): None,  # not swept yet
}


def lr_for(controls, method, where):
    """The lr a training config gets: the default for (backend, method) when the controls give none;
    an lr that differs from the default (or has none to differ from) must carry controls.lr_reason."""
    backend = controls.get("student_backend", "hf_lora")
    default = LR_DEFAULTS.get((backend, method))
    lr = controls.get("lr")
    if lr is None:
        if default is None:
            raise ValueError(f"{where}: no default lr for ({backend}, {method}); set controls.lr and controls.lr_reason")
        return default
    if lr != default and not controls.get("lr_reason"):
        raise ValueError(f"{where}: lr {lr} differs from the default {default} for ({backend}, {method}) "
                         "(mopd.training.controls.LR_DEFAULTS); add controls.lr_reason")
    return lr


def check(controls, where):
    unknown = sorted(set(controls) - CONTROL_KEYS)
    if unknown:
        raise ValueError(f"{where}: unknown control keys {unknown} (known: mopd.training.controls.CONTROL_KEYS)")


def training_fields(controls, *, micro_batch_size, method="sft", where="controls"):
    """Common fields of every training config built from these controls, in config order.
    method: "sft" | "mopd" picks the default lr (lr_for); None (merges) takes the controls' lr as is."""
    c = controls
    return {
        "student": c["student"],
        "student_backend": c.get("student_backend", "hf_lora"),
        "student_prompt_format": c.get("student_prompt_format", "plain_qa_v1"),
        "seed": c["seed"],
        "steps": c["steps"],
        "lr": lr_for(c, method, where) if method else c.get("lr"),
        "lr_schedule": c["lr_schedule"],
        "micro_batch_size": micro_batch_size,
        "save_every": c["save_every"],
    }


def system_fields(controls):
    """Training-system settings present in the controls (results unchanged, speed/memory differ)."""
    return {k: controls[k] for k in SYSTEM_KEYS if k in controls}
