"""Canonical manifest for generated benchmark outputs across inference backends."""

import hashlib
import json


def build_generation_manifest(
    *,
    run_id,
    model_version_id,
    protocol_id,
    prompt_mode,
    generation,
    data_path,
    data_sha256,
    backend,
    weight_representation,
    inference_dtype,
    model,
):
    protocol = {
        "protocol_id": protocol_id,
        "prompt_mode": prompt_mode,
        "generation": generation,
    }
    canonical = json.dumps(protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "schema": "ifairy.benchmark_generation.v1",
        "run_id": run_id,
        "model_version_id": model_version_id,
        "protocol_id": protocol_id,
        "protocol_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "protocol": protocol,
        "data_path": str(data_path),
        "data_sha256": data_sha256,
        "dataset": {"path": str(data_path), "sha256": data_sha256},
        "backend": backend,
        "weight_representation": weight_representation,
        "inference_dtype": inference_dtype,
        "model": model,
        "generation": generation,
    }
