"""Write a stable model-version manifest for each MOPD run and checkpoint."""
import hashlib
import json
import os
from pathlib import Path


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _file_identity(path: str | Path) -> dict:
    source = Path(path).resolve()
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {"path": str(source), "sha256": digest.hexdigest(), "size_bytes": size}


def _build_training_inputs(config: dict, run: Path) -> dict:
    config_path = run / "config.json"
    config_identity = _file_identity(config_path)
    domains = {}
    for name, spec in config.get("domains", {}).items():
        train_file = spec.get("train_file")
        sources = spec.get("sources") or ([{"name": "train", "path": train_file, "weight": 1.0}] if train_file else [])
        domains[name] = {
            "train": _file_identity(sources[0]["path"]) if len(sources) == 1 else None,
            "sources": [{"name": s["name"], "weight": s["weight"],
                         "per_step": spec.get("source_counts", {}).get(s["name"]), **_file_identity(s["path"])}
                        for s in sources],
            "mixture": spec.get("mixture_preset"),
            "holdouts": [_file_identity(path) for path in spec.get("holdout_files", [])],
        }
    if not domains and config.get("train_file"):
        domains["default"] = {
            "train": _file_identity(config["train_file"]),
            "holdouts": [_file_identity(path) for path in config.get("holdout_files", [])],
        }
    return {
        "schema": "ifairy.training_inputs.v1",
        "config": config_identity,
        "domains": domains,
    }


def _teacher_source(path):
    """sft_expert (with its adapter identity) when exported by scripts/export_lora_expert.py,
    otherwise an external checkpoint such as an official post-trained model."""
    record = Path(path or "") / "expert-source.json"
    if path and record.is_file():
        info = json.loads(record.read_text())
        return {"kind": "sft_expert", "adapter": info["adapter"], "adapter_sha256": info["adapter_sha256"]}
    return {"kind": "external"}


def build_manifest(
    config: dict,
    run_dir: str | Path,
    status: dict | None = None,
    training_inputs: dict | None = None,
) -> dict:
    run = Path(run_dir).resolve()
    status = status or {}
    if training_inputs is None:
        inputs_path = run / "training-inputs.json"
        if inputs_path.is_file():
            training_inputs = json.loads(inputs_path.read_text())
    checkpoints = []
    for checkpoint in sorted(run.glob("checkpoint-*")):
        if not checkpoint.is_dir():
            continue
        suffix = checkpoint.name.removeprefix("checkpoint-")
        try:
            step = int(suffix)
        except ValueError:
            step = None
        if (checkpoint / "qat_scales.safetensors").is_file():
            artifact_type = "fairy2i_qat_scale_update"
            complete = (checkpoint / "qat-update.json").is_file()
        else:
            artifact_type = ("lora_adapter" if (checkpoint / "adapter_config.json").is_file()
                             else "full_model" if (checkpoint / "config.json").is_file() else "checkpoint")
            complete = (checkpoint / "adapter_model.safetensors").is_file() or any(checkpoint.glob("*.safetensors"))
        checkpoints.append({
            "version_id": f"{run.name}:step-{suffix}",
            "path": str(checkpoint),
            "step": step,
            "artifact_type": artifact_type,
            "complete": complete,
        })

    student_backend = config.get("student_backend", "hf_lora")
    student_precision = config.get("forward_precision", config.get("student_precision", config.get("dtype", "bfloat16")))
    quantization = config.get("quantization", config.get("student_quantization"))
    if student_backend == "fairy2i_qat" and quantization is None:
        quantization = {"scheme": "fairy2i_complex_1bit", "scale_update": "trainable", "base_weights": "frozen_bfloat16"}
    student = {
        "base_model": config.get("student"),
        "backend": student_backend,
        "update_type": config.get("update_type", {"fairy2i_qat": "quant_scale_only", "hf_full": "full"}.get(student_backend, "lora")),
        "initial_adapter": config.get("student_adapter"),
        "prompt_format": config.get("student_prompt_format"),
        "forward_precision": student_precision,
        "quantization": quantization,
    }
    teachers = {
        domain: {"model": spec.get("teacher"), "device": spec.get("device"), "trainable": False,
                 "prompt_format": spec.get("teacher_prompt_format", "chat_template_v1"),
                 "source": _teacher_source(spec.get("teacher"))}
        for domain, spec in config.get("domains", {}).items()
    }
    if not teachers and config.get("teacher"):
        teachers["default"] = {"model": config["teacher"], "device": config.get("teacher_device"), "trainable": False}

    training = {
        "method": config.get("training_method", "mopd"),
        "seed": config.get("seed"),
        "planned_steps": config.get("steps"),
        "completed_steps": status.get("step", status.get("steps")),
        "state": status.get("state", "unknown"),
        "config_path": str(run / "config.json"),
    }
    if training_inputs is not None:
        training["config_sha256"] = training_inputs["config"]["sha256"]
        training["inputs"] = training_inputs

    return {
        "schema": "ifairy.model_version.v1",
        "version_id": run.name,
        "run_id": run.name,
        "run_path": str(run),
        "training": training,
        "components": {"student": student, "teachers": teachers},
        "rollout": {
            "backend": config.get("rollout_backend", "hf"),
            "devices": config.get("rollout_devices", []),
        },
        "checkpoints": checkpoints,
    }


def write_manifest(config: dict, run_dir: str | Path, status: dict | None = None) -> Path:
    run = Path(run_dir).resolve()
    inputs_path = run / "training-inputs.json"
    if inputs_path.is_file():
        training_inputs = json.loads(inputs_path.read_text())
        current_config = _file_identity(run / "config.json")
        if current_config["sha256"] != training_inputs["config"]["sha256"]:
            raise ValueError("Run config changed after its model-version inputs were recorded")
    else:
        training_inputs = _build_training_inputs(config, run)
        _atomic_json(inputs_path, training_inputs)
    path = run / "model-version.json"
    _atomic_json(path, build_manifest(config, run, status, training_inputs))
    return path


def resolve_checkpoint(manifest: dict, reference: str) -> dict:
    """Resolve `base` or one complete checkpoint from a model-version manifest."""
    run_id = manifest.get("run_id") or manifest.get("version_id")
    student = manifest.get("components", {}).get("student", {})
    if reference == "base":
        base_model = student.get("base_model")
        if not base_model:
            raise ValueError("Model-version manifest has no student base_model")
        return {
            "version_id": f"{run_id}:base",
            "path": None,
            "artifact_type": "base_weights",
            "complete": True,
            "base_model": base_model,
        }

    matches = [
        item for item in manifest.get("checkpoints", [])
        if reference in (item.get("version_id"), item.get("path"), Path(item.get("path", "")).name)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one checkpoint matching {reference!r}; found {len(matches)}")
    checkpoint = matches[0]
    if checkpoint.get("complete") is not True:
        raise ValueError(f"Checkpoint is incomplete: {checkpoint.get('version_id', reference)}")
    if not checkpoint.get("path") or not checkpoint.get("version_id"):
        raise ValueError("Checkpoint entry must include path and version_id")
    return {**checkpoint, "base_model": student.get("base_model")}
