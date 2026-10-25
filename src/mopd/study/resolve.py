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


def run_config(study, name, fields, evals, spec_path, spec_sha, reg):
    """The SFT training config of one run (same fields as scripts/make_experiment.py writes)."""
    f = {**DEFAULTS, **fields}
    if f["method"] != "sft":
        raise NotImplementedError(f"{name}: method {f['method']!r} (studies start with SFT)")
    domains = f.get("domains") or [f["domain"]]
    n = f["samples_per_step"]
    counts = [n // len(domains) + (i < n % len(domains)) for i in range(len(domains))]
    eval_data = [reg["evals"][e]["data"] for e in evals]
    used, dspec = [], {}
    for d, count in zip(domains, counts):
        ds = reg["data"][study["data"][d]]
        holdout = list(dict.fromkeys(ds.get("holdout", []) + eval_data))
        used += [ds["train"], *holdout]
        dspec[d] = {"train_file": abs_path(ds["train"]), "holdout_files": [abs_path(h) for h in holdout],
                    "count": count, "weight": 1 / len(domains)}
    model_rel = reg["models"][f["student"]]["path"]
    ctl = {"student": abs_path(model_rel), **{k: f[k] for k in f if k in controls.CONTROL_KEYS and k != "student"}}
    if "lr" in (study.get("axes") or {}):
        ctl["lr_reason"] = f"lr axis of study {study['study']}"
    cfg = {
        **controls.training_fields(ctl, micro_batch_size=f["micro_batch_size"], method="sft", where=f"{study['study']}/{name}"),
        "output": abs_path(f"runs/studies/{study['study']}/{name}"),
        "data_pins": {abs_path(r): pin(reg, r) for r in dict.fromkeys(used)} | {abs_path(k): v for k, v in model_pins(reg, model_rel).items()},
        "experiment": {"name": study["study"], "arm": name, "spec_path": spec_path, "spec_sha256": spec_sha},
        **controls.system_fields(ctl),
        "training_method": "sft",
        "student_device": "cuda:0",
        "max_seq_len": f["sft_max_seq_len"],
        "domains": dspec,
    }
    return cfg


def identity(cfg):
    """sha256 (16 hex) of the fields that change the trained weights; works on any SFT config,
    including older experiments' (their data_pins hold the needed sha)."""
    pins = cfg["data_pins"]
    student = sorted(v for k, v in pins.items() if k.startswith(cfg["student"] + "/") and k.endswith(".safetensors"))
    key = {
        **{k: cfg.get(k) for k in IDENTITY_KEYS},
        "student": student or cfg["student"],
        "domains": {d: {"train": pins.get(s["train_file"], s["train_file"]), "count": s["count"], "weight": round(s["weight"], 6)}
                    for d, s in sorted(cfg["domains"].items())},
    }
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
