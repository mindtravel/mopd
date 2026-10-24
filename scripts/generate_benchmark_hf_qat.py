"""Generate resumable benchmark outputs with the native Fairy2i QAT forward."""

import argparse
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path("/ssd/lzx/mopd")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def emit_progress(payload):
    print(json.dumps(payload), flush=True)


def load_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    keys = [(row["domain"], str(row["id"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Benchmark input has duplicate (domain, id) keys")
    if any(domain not in {"math", "instruction"} for domain, _ in keys):
        raise ValueError("Benchmark input contains an unsupported domain")
    return rows, set(keys)


def resume_records(path, expected_keys):
    path = Path(path)
    if not path.exists():
        return set()
    done = set()
    with path.open("r+b") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if stream.read():
                    raise ValueError("Corruption before final generated record")
                stream.truncate(offset)
                break
            key = (row["domain"], str(row["id"]))
            if key not in expected_keys or key in done:
                raise ValueError(f"Duplicate or foreign generated record: {key}")
            done.add(key)
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--checkpoint", required=True, help="base or a checkpoint version ID/path")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, help="default: max(16384, max-new-tokens+4096)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261014)
    from mopd.data.templates import STUDENT_PROMPT_MODES

    parser.add_argument("--student-prompt-format", required=True, choices=sorted(STUDENT_PROMPT_MODES))
    args = parser.parse_args()
    if args.max_model_len is None:
        from mopd.evaluation.context import default_max_model_len

        args.max_model_len = default_max_model_len(args.max_new_tokens)

    manifest_path = Path(args.model_version).resolve()
    data_path = Path(args.data).resolve()
    output = Path(args.out).resolve()
    root = ROOT.resolve()
    if not manifest_path.is_relative_to(root) or manifest_path.name != "model-version.json":
        raise ValueError("Model-version manifest must be inside /ssd/lzx/mopd")
    run_dir = manifest_path.parent
    if not data_path.is_relative_to(root) or not output.is_relative_to(run_dir):
        raise ValueError("Benchmark data and output must stay inside the MOPD project/run")
    if args.batch_size < 1 or args.max_new_tokens < 1 or args.max_model_len < args.max_new_tokens:
        raise ValueError("Invalid generation limits")

    from mopd.evaluation.versioning import resolve_checkpoint

    model_manifest = json.loads(manifest_path.read_text())
    student = model_manifest["components"]["student"]
    if student.get("backend") != "fairy2i_qat":
        raise ValueError("This generator only supports the native Fairy2i QAT forward")
    checkpoint = resolve_checkpoint(model_manifest, args.checkpoint)
    update_path = checkpoint["path"]
    if update_path and not Path(update_path).resolve().is_relative_to(run_dir.resolve()):
        raise ValueError("Checkpoint path is outside its model-version run")
    if update_path and not (Path(update_path) / "qat_scales.safetensors").is_file():
        raise FileNotFoundError("Fairy2i checkpoint has no qat_scales.safetensors")

    from mopd.training.batching import normalize_eos_token_ids

    base_path = Path(student["base_model"])
    generation_path = base_path / "generation_config.json"
    if generation_path.is_file():
        base_generation = json.loads(generation_path.read_text())
        eos_ids = normalize_eos_token_ids(base_generation.get("eos_token_id"))
    else:
        eos_ids = ()
    if not eos_ids:
        base_config = json.loads((base_path / "config.json").read_text())
        eos_ids = normalize_eos_token_ids(base_config.get("eos_token_id"))
    if not eos_ids:
        raise ValueError("Student model has no configured EOS token IDs")

    data_sha256 = sha256_file(data_path)
    generation_options = {
        "temperature": 0,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "seed": args.seed,
        "eos_token_ids": list(eos_ids),
    }
    from mopd.evaluation.generation_manifest import build_generation_manifest

    generation = build_generation_manifest(
        run_id=model_manifest["run_id"],
        model_version_id=checkpoint["version_id"],
        protocol_id=args.protocol_id,
        prompt_mode=STUDENT_PROMPT_MODES[args.student_prompt_format],
        generation=generation_options,
        data_path=data_path,
        data_sha256=data_sha256,
        backend="hf_fairy2i_qat",
        weight_representation="fairy2i_complex_1bit_with_scale_update" if update_path else "fairy2i_complex_1bit_base",
        inference_dtype="bfloat16_activations_custom_1bit_linear",
        model={
            "base_model": student["base_model"],
            "checkpoint_path": update_path,
            "checkpoint_artifact_type": checkpoint["artifact_type"],
            "checkpoint_sha256": sha256_file(Path(update_path) / "qat_scales.safetensors") if update_path else None,
        },
    )

    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "generation.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Generation already running: {lock_path}") from error
        config_path = output / "generation-config.json"
        if config_path.exists():
            if json.loads(config_path.read_text()) != generation:
                raise ValueError("Resume configuration mismatch")
        else:
            atomic_json(config_path, generation)

        rows, expected_keys = load_rows(data_path)
        records_path = output / "generated.jsonl"
        done = resume_records(records_path, expected_keys)
        pending = [row for row in rows if (row["domain"], str(row["id"])) not in done]
        if not pending:
            return

        import torch
        from mopd.backends.fairy2i_qat_adapter import load_fairy2i_qat_student
        from mopd.data.domains import student_prompt

        torch.manual_seed(args.seed)
        tokenizer, model = load_fairy2i_qat_student(
            student["base_model"], args.device, trainable_scales=False, update_path=update_path
        )
        from mopd.training.batching import generation_eos_ids, is_eos_token, trim_generated

        if generation_eos_ids(model, tokenizer) != eos_ids:
            raise ValueError("Loaded student EOS tokens differ from the recorded generation protocol")
        model.eval()
        started = time.monotonic()
        with records_path.open("a", encoding="utf-8", buffering=1) as stream:
            for offset in range(0, len(pending), args.batch_size):
                batch = pending[offset : offset + args.batch_size]
                prompts = [student_prompt(tokenizer, row, args.student_prompt_format) for row in batch]
                encoded = tokenizer(prompts, padding=True, return_tensors="pt").to(args.device)
                if encoded.input_ids.shape[1] + args.max_new_tokens > args.max_model_len:
                    raise ValueError(f"Prompt plus output limit exceeds context at ID {batch[0]['id']}")
                with torch.inference_mode():
                    output_ids = model.generate(
                        **encoded,
                        do_sample=False,
                        max_new_tokens=args.max_new_tokens,
                        eos_token_id=list(eos_ids),
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        use_cache=True,
                    )
                for index, (source, sequence) in enumerate(zip(batch, output_ids)):
                    generated_ids = sequence[encoded.input_ids.shape[1] :]
                    token_ids = trim_generated(generated_ids, eos_ids)
                    stopped = bool(token_ids) and is_eos_token(token_ids[-1], eos_ids)
                    if stopped:
                        generated_ids = generated_ids[: len(token_ids)]
                    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    record = {
                        "id": source["id"],
                        "domain": source["domain"],
                        "prompt": source["prompt"],
                        "response": response,
                        "response_ids": token_ids,
                        "tokens": len(token_ids),
                        "truncated": not stopped and len(token_ids) >= args.max_new_tokens,
                        "finish_reason": "stop" if stopped else "length",
                        "model_version_id": checkpoint["version_id"],
                        "protocol_id": args.protocol_id,
                    }
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    done.add((record["domain"], str(record["id"])))
                stream.flush()
                os.fsync(stream.fileno())
                atomic_json(
                    output / "generation-progress.json",
                    {
                        "done": len(done),
                        "total": len(rows),
                        "seconds_this_session": time.monotonic() - started,
                        "backend": "hf_fairy2i_qat",
                        "model_version_id": checkpoint["version_id"],
                        "protocol_id": args.protocol_id,
                    },
                )
                emit_progress({"done": len(done), "total": len(rows)})


if __name__ == "__main__":
    main()
