"""One run of a study -> the training config the trainers read, its identity and its directory.

Identity is the sha256 of what changes the trained weights: the student (by the sha of its weight
files), backend, prompt format, seed, steps, lr and schedule, sequence cap, method, and each domain's
training file (by sha), count and weight. Paths, checkpoint cadence, micro-batching, FSDP settings
and holdout files (a check, not a filter) are left out, so a finished run with the same identity --
from any study or older experiment -- can be reused instead of retrained.
"""

import hashlib
import json
from pathlib import Path

import yaml

from mopd import paths
from mopd.training import controls

ROOT_CODE = Path(__file__).resolve().parents[3]
REGISTRY = ROOT_CODE / "configs/registry.yaml"

# What a study does not have to write (full256's settings; lr from controls.LR_DEFAULTS)
DEFAULTS = {
    "method": "sft",
    "student": "qwen3-4b-base",
    "student_backend": "hf_full",
    "student_prompt_format": "plain_qa_v1",
    "seed": 20261014,
    "steps": 256,
    "samples_per_step": 32,
    "lr_schedule": {"type": "constant"},
    "micro_batch_size": 4,
    "save_every": 64,
    "sft_max_seq_len": 10240,
    "fsdp": {"reshard_after_forward": True},
}
# MOPD (scripts/train_multidomain_opd.py): the settings measured to fit and run on the 0-3 layout
# (2026-10-25 bench: student on 2 GPUs with offload, teachers on the student GPUs, rollout on the rest)
MOPD_DEFAULTS = {
    "micro_batch_size": 16,
    "micro_batch_tokens": 8192,
    "fsdp": {"reshard_after_forward": True, "offload": True},
    "mopd_max_new_tokens": 8192,
    "mopd_generation_batch_size": 8,
    "mopd_vllm_memory_utilization": 0.8,
    "teacher_prompt_format": "student_prefix_v1",
    "mopd_rollout_async": True,
}
MOPD_IDENTITY_KEYS = ("max_new_tokens", "rollout_async")
IDENTITY_KEYS = ("student_backend", "student_prompt_format", "seed", "steps", "lr", "lr_schedule",
                 "training_method", "max_seq_len")


def registry():
    return yaml.safe_load(REGISTRY.read_text())


def abs_path(rel):
    return f"{paths.ROOT}/{rel}"


def pin(reg, rel):
    if rel not in reg["pins"]:
        raise ValueError(f"{rel} is not pinned in {REGISTRY.name} (python scripts/study.py pin)")
    return reg["pins"][rel]


def model_pins(reg, model_rel):
    return {k: v for k, v in reg["pins"].items() if k.startswith(model_rel + "/")}


def layout(n_domains, offload):
    """MOPD devices from the training GPUs (paths.GPU_ROLES): with offload the student takes two, the
    teachers share the student GPUs (their memory is free once the student state is in host
    memory), and every other training GPU runs a rollout engine."""
    gpus = list(range(len(paths.TRAIN_GPUS)))  # logical: the pool launches a job on CUDA_VISIBLE_DEVICES = its GPUs
    if not offload or len(gpus) < 3:
        raise ValueError("MOPD on the training GPUs needs fsdp.offload (student on 2 GPUs) and a third GPU for rollouts")
    student = gpus[:2]
    return {"student": student, "teachers": [student[i % 2] for i in range(n_domains)], "rollout": gpus[2:]}


def run_config(study, name, fields, evals, spec_path, spec_sha, reg, teachers=None):
    """The training config of one run (same fields as scripts/make_experiment.py writes).
    teachers: {domain: checkpoint directory} for method mopd (resolved by the runner)."""
    method = {**DEFAULTS, **fields}["method"]
    domains = fields.get("domains") or [fields["domain"]]
    # `dataset` (an axis over datasets of one domain) overrides the study's data map
    datasets = {d: fields.get("dataset") if len(domains) == 1 and fields.get("dataset") else study["data"][d] for d in domains}
    own = reg["data"][datasets[domains[0]]].get("train_settings", {}) if len(domains) == 1 else {}
    f = {**DEFAULTS, **(MOPD_DEFAULTS if method == "mopd" else {}), **own, **fields}
    if f["method"] not in ("sft", "mopd"):
        raise NotImplementedError(f"{name}: method {f['method']!r}")
    n = f["samples_per_step"]
    counts = [n // len(domains) + (i < n % len(domains)) for i in range(len(domains))]
    eval_data = [reg["evals"][e]["data"] for e in evals]
    used, dspec = [], {}
    for d, count in zip(domains, counts):
        ds = reg["data"][datasets[d]]
        holdout = list(dict.fromkeys(ds.get("holdout", []) + eval_data))
        used += [ds["train"], *holdout]
        dspec[d] = {"train_file": abs_path(ds["train"]), "holdout_files": [abs_path(h) for h in holdout],
                    "count": count, "weight": 1 / len(domains)}
    model_rel = reg["models"][f["student"]]["path"]
    ctl = {"student": abs_path(model_rel), **{k: f[k] for k in f if k in controls.CONTROL_KEYS and k != "student"}}
    if ctl.get("student_backend") != "hf_full":
        ctl.pop("fsdp", None)  # a LoRA student trains on one GPU, unsharded
    # a dataset's micro_batch_tokens is a training-system setting like the others in SYSTEM_KEYS
    if "lr" in (study.get("axes") or {}):
        ctl["lr_reason"] = f"lr axis of study {study['study']}"
    cfg = {
        **controls.training_fields(ctl, micro_batch_size=f["micro_batch_size"], method=f["method"], where=f"{study['study']}/{name}"),
        "output": abs_path(f"runs/studies/{study['study']}/{name}"),
        "data_pins": {abs_path(r): pin(reg, r) for r in dict.fromkeys(used)} | {abs_path(k): v for k, v in model_pins(reg, model_rel).items()},
        "experiment": {"name": study["study"], "arm": name, "spec_path": spec_path, "spec_sha256": spec_sha},
        **controls.system_fields(ctl),
    }
    if f["method"] == "sft":
        return {**cfg, "training_method": "sft", "student_device": "cuda:0", "max_seq_len": f["sft_max_seq_len"], "domains": dspec}
    dev = layout(len(domains), f["fsdp"].get("offload"))
    for d, gpu in zip(domains, dev["teachers"]):
        dspec[d] = {"teacher": str(teachers[d]), "device": f"cuda:{gpu}", "teacher_prompt_format": f["teacher_prompt_format"], **dspec[d]}
    return {
        **cfg,
        "training_method": "mopd",
        "student_device": f"cuda:{dev['student'][0]}",
        "domains": dspec,
        "student_gpus": len(dev["student"]),
        "max_new_tokens": f["mopd_max_new_tokens"],
        "generation_batch_size": f["mopd_generation_batch_size"],
        "rollout_backend": "vllm",
        "rollout_devices": [f"cuda:{g}" for g in dev["rollout"]],
        "vllm_memory_utilization": f["mopd_vllm_memory_utilization"],
        **({"rollout_async": True, "rollout_streaming": True} if f.get("mopd_rollout_async") else {}),
    }


def identity(cfg):
    """sha256 (16 hex) of the fields that change the trained weights; works on any SFT config,
    including older experiments' (their data_pins hold the needed sha)."""
    pins = cfg["data_pins"]
    student = sorted(v for k, v in pins.items() if k.startswith(cfg["student"] + "/") and k.endswith(".safetensors"))
    key = {
        **{k: cfg.get(k) for k in IDENTITY_KEYS},
        **({k: cfg.get(k) for k in MOPD_IDENTITY_KEYS} if cfg.get("training_method") == "mopd" else {}),
        **({"lora": cfg["lora"]} if cfg.get("lora") else {}),  # rank / targets change what is trained
        "student": student or cfg["student"],
        "domains": {d: {"train": pins.get(s["train_file"], s["train_file"]), "count": s["count"], "weight": round(s["weight"], 6),
                        **({k: s[k] for k in ("teacher", "teacher_prompt_format") if k in s})}
                    for d, s in sorted(cfg["domains"].items())},
    }
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
