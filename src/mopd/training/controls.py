"""The controlled training fields of a comparison, shared by experiment specs
(scripts/make_experiment.py) and sweep specs (scripts/run_sweep.py): one list of known control keys
(a typo is an error, not a silently ignored setting) and one function that turns controls into a
training config's common fields, in a fixed order (generated configs are committed and checked
byte for byte)."""

CONTROL_KEYS = {
    # every comparison
    "student", "student_backend", "student_prompt_format", "seed", "steps", "samples_per_step", "lr",
    "lr_schedule", "save_every",
    # training system (same result, different speed/memory): mopd.training.fsdp, students.registry
    "fsdp", "gradient_checkpointing", "lora", "micro_batch_tokens",
    # experiment specs
    "micro_batch_size", "sft_max_seq_len", "mopd_max_new_tokens", "mopd_generation_batch_size",
    "mopd_vllm_memory_utilization", "teacher_prompt_format",
}
SYSTEM_KEYS = ("fsdp", "gradient_checkpointing", "lora", "micro_batch_tokens")


def check(controls, where):
    unknown = sorted(set(controls) - CONTROL_KEYS)
    if unknown:
        raise ValueError(f"{where}: unknown control keys {unknown} (known: mopd.training.controls.CONTROL_KEYS)")


def training_fields(controls, *, micro_batch_size):
    """Common fields of every training config built from these controls, in config order."""
    c = controls
    return {
        "student": c["student"],
        "student_backend": c.get("student_backend", "hf_lora"),
        "student_prompt_format": c.get("student_prompt_format", "plain_qa_v1"),
        "seed": c["seed"],
        "steps": c["steps"],
        "lr": c["lr"],
        "lr_schedule": c["lr_schedule"],
        "micro_batch_size": micro_batch_size,
        "save_every": c["save_every"],
    }


def system_fields(controls):
    """Training-system settings present in the controls (results unchanged, speed/memory differ)."""
    return {k: controls[k] for k in SYSTEM_KEYS if k in controls}
