"""Dispatch model-version evaluation to a compatible generation backend and scorer."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path("/ssd/lzx/mopd")


def build_command(args):
    manifest_path = Path(args.model_version).resolve()
    data_path = Path(args.data).resolve()
    if not manifest_path.is_relative_to(ROOT) or manifest_path.name != "model-version.json":
        raise ValueError("Model-version manifest must be inside /ssd/lzx/mopd")
    if not data_path.is_relative_to(ROOT):
        raise ValueError("Benchmark data must be inside /ssd/lzx/mopd")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.protocol_id):
        raise ValueError("protocol-id may contain only letters, digits, dot, underscore, and dash")

    from mopd.evaluation.versioning import resolve_checkpoint

    model_manifest = json.loads(manifest_path.read_text())
    student = model_manifest["components"]["student"]
    checkpoint = resolve_checkpoint(model_manifest, args.checkpoint)
    requested = args.backend
    if requested == "auto":
        requested = "hf_qat" if student.get("backend") == "fairy2i_qat" else "vllm"
    if requested == "hf_qat" and student.get("backend") != "fairy2i_qat":
        raise ValueError("hf_qat backend requires a Fairy2i QAT student")
    if requested == "vllm" and checkpoint["artifact_type"] not in {"base_weights", "lora_adapter", "full_model"}:
        raise ValueError(f"vLLM cannot load checkpoint artifact {checkpoint['artifact_type']!r}")
    if checkpoint["artifact_type"] == "full_model":
        from mopd.runtime.model_files import check_config_matches_base

        check_config_matches_base(checkpoint["path"], student["base_model"])
    if requested == "hf_qat" and checkpoint["artifact_type"] not in {"base_weights", "fairy2i_qat_scale_update"}:
        raise ValueError(f"HF QAT cannot load checkpoint artifact {checkpoint['artifact_type']!r}")

    # The student must be prompted the way it was trained; legacy manifests predate the field.
    recorded_format = student.get("prompt_format")
    requested_format = getattr(args, "student_prompt_format", None)
    if recorded_format and requested_format and requested_format != recorded_format:
        raise ValueError(f"Run was trained with student prompt {recorded_format!r}, not {requested_format!r}")
    prompt_format = recorded_format or requested_format
    if not prompt_format:
        raise ValueError("Model-version manifest has no student prompt_format; pass --student-prompt-format")

    run_dir = manifest_path.parent
    checkpoint_dir = Path(checkpoint["path"]).name if checkpoint["path"] else "base"
    output = Path(args.out).resolve() if args.out else run_dir / "evaluations" / checkpoint_dir / args.protocol_id
    if not output.is_relative_to(run_dir.resolve()):
        raise ValueError("Evaluation output must remain inside the model-version run")

    if requested == "hf_qat":
        generator = ROOT / "scripts/generate_benchmark_hf_qat.py"
        command = [
            str(ROOT / "mopd_train/bin/python"),
            str(generator),
            "--model-version",
            str(manifest_path),
            "--checkpoint",
            args.checkpoint,
            "--data",
            str(data_path),
            "--out",
            str(output),
            "--device",
            "cuda:0",
            "--protocol-id",
            args.protocol_id,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--max-model-len",
            str(args.max_model_len),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(args.seed),
            "--student-prompt-format",
            prompt_format,
        ]
        env = os.environ.copy()
        if len(args.device) != 1:
            raise ValueError("the hf_qat backend runs on one GPU")
        env["CUDA_VISIBLE_DEVICES"] = str(args.device[0])
        env["PYTHONPATH"] = str(ROOT / "src") + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["HF_HOME"] = str(ROOT / "cache/huggingface")
        env["NLTK_DATA"] = str(ROOT / "cache/nltk")
    else:
        generator = ROOT / "scripts/generate_benchmark_vllm.py"
        command = [
            str(ROOT / "mopd_infer/bin/python"),
            str(generator),
            # a fully fine-tuned checkpoint is itself the model; a LoRA one is an adapter on the base
            "--model",
            checkpoint["path"] if checkpoint["artifact_type"] == "full_model" else student["base_model"],
            "--data",
            str(data_path),
            "--out",
            str(output),
            "--device",
            *map(str, args.device),
            "--model-version-id",
            checkpoint["version_id"],
            "--protocol-id",
            args.protocol_id,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--max-seqs",
            str(args.max_seqs),
            "--batch-size",
            str(args.batch_size),
            "--max-num-batched-tokens",
            str(args.max_num_batched_tokens),
            "--max-model-len",
            str(args.max_model_len),
            "--student-prompt-format",
            prompt_format,
        ]
        if checkpoint["path"] and checkpoint["artifact_type"] == "lora_adapter":
            command.extend(["--adapter", checkpoint["path"]])
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src") + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    return command, env, output, requested


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--checkpoint", required=True, help="base or a checkpoint version ID/path")
    parser.add_argument("--data", required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--out")
    parser.add_argument("--backend", choices=("auto", "hf_qat", "vllm"), default="auto")
    parser.add_argument(
        "--device", type=int, nargs="+", required=True, help="physical GPU ID(s); vLLM spreads rows over all of them"
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, help="default: max(16384, max-new-tokens+4096)")
    parser.add_argument("--max-seqs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048, help="vLLM prefill token budget")
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--student-prompt-format", help="required only for manifests without student prompt_format")
    parser.add_argument("--skip-grade", action="store_true", help="generate only; grade later on CPU (frees the GPUs)")
    args = parser.parse_args()
    if args.max_model_len is None:
        from mopd.evaluation.context import default_max_model_len

        args.max_model_len = default_max_model_len(args.max_new_tokens)

    command, env, output, backend = build_command(args)
    output.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"backend": backend, "output": str(output), "command": command}, ensure_ascii=False), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    if args.skip_grade:
        return
    scorer = [
        str(ROOT / "mopd_train/bin/python"),
        str(ROOT / "scripts/grade_benchmark_outputs.py"),
        "--data",
        str(Path(args.data).resolve()),
        "--out",
        str(output),
    ]
    subprocess.run(scorer, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
