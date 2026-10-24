"""Resumable benchmark generation only; grading runs in the training environment.

Streams: every finished row is written to generated.jsonl and generation-progress.json the moment
it completes (mopd.rollout.engine_stream), not after the whole submitted set finishes -- the same
technique MOPD's training rollout uses, so a long run (long outputs, large benchmarks) is visibly
making progress instead of looking stuck until it writes everything at once.

Several GPUs (--device 0 1 2 ...): one resident engine per GPU (mopd.rollout.engine_pool) pulls rows
whenever its scheduler has room, so no GPU idles while another still holds a backlog; this process
alone writes generated.jsonl, so output and resume are the same for any GPU count."""

from mopd import paths
import argparse, collections, fcntl, hashlib, json, os, signal, time
from pathlib import Path
from mopd.evaluation.generation_manifest import build_generation_manifest
from mopd.data.templates import STUDENT_PROMPT_MODES
from mopd.runtime import progress
from mopd.rollout.engine_pool import EnginePool


def main():
    p = argparse.ArgumentParser()
    for name in ("model", "data", "out"):
        p.add_argument("--" + name, required=True)
    p.add_argument(
        "--device", type=int, nargs="+", required=True, help="one or more GPUs; rows are handed to whichever has room"
    )
    p.add_argument("--adapter")
    p.add_argument("--teacher", action="store_true")
    p.add_argument("--model-version-id")
    p.add_argument("--protocol-id")
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--max-seqs", type=int, default=32)
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="vestigial: every pending row is submitted "
        "to the engine at once regardless (streaming makes chunking for progress visibility pointless); "
        "kept so existing callers that pass it do not need to change",
    )
    p.add_argument("--student-prompt-format", choices=sorted(STUDENT_PROMPT_MODES))
    # vLLM prefill token budget per scheduler step. 2048 reproduces every earlier evaluation's runtime;
    # with --batch-size >= rows and a large --max-seqs, the whole set goes through continuous batching at once.
    p.add_argument("--max-num-batched-tokens", type=int, default=2048)
    p.add_argument(
        "--temperature", type=float, default=0.0, help="0 = greedy; long-reasoning students loop under greedy, use ~0.6"
    )
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-model-len", type=int, help="default: max(16384, max-new-tokens+4096)")
    a = p.parse_args()
    from mopd.evaluation.context import default_max_model_len

    if a.max_model_len is None:
        a.max_model_len = default_max_model_len(a.max_new_tokens)
    if not a.teacher and not a.student_prompt_format:
        p.error("--student-prompt-format is required for student generation")
    out = Path(os.path.abspath(a.out))  # symlinked dirs inside the project count as inside
    root = paths.ROOT
    if not out.is_relative_to(root):
        raise ValueError("Output outside project")
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "generation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    data_sha256 = hashlib.sha256(Path(a.data).read_bytes()).hexdigest()
    model_version_id = a.model_version_id
    if not model_version_id and a.adapter:
        adapter_path = Path(a.adapter).resolve()
        manifest_path = adapter_path.parent / "model-version.json"
        if manifest_path.is_file():
            version = json.loads(manifest_path.read_text())
            match = next(
                (item for item in version.get("checkpoints", []) if Path(item["path"]).resolve() == adapter_path), None
            )
            model_version_id = match["version_id"] if match else version.get("version_id")
    if not model_version_id:
        model_version_id = ("teacher:" if a.teacher else "base:") + Path(a.model).name
    protocol_id = a.protocol_id or f"mopd-{Path(a.data).stem}-greedy-{a.max_new_tokens}-v1"
    generation_options = {
        "temperature": 0,
        "max_new_tokens": a.max_new_tokens,
        "max_model_len": a.max_model_len,
        "seed": 20261013,
    }
    if a.temperature > 0:
        generation_options.update(temperature=a.temperature, top_p=a.top_p)  # greedy configs stay byte-identical
    adapter_sha256 = (
        hashlib.sha256((Path(a.adapter) / "adapter_model.safetensors").read_bytes()).hexdigest() if a.adapter else None
    )
    config = build_generation_manifest(
        run_id=model_version_id.rsplit(":step-", 1)[0] if ":step-" in model_version_id else model_version_id,
        model_version_id=model_version_id,
        protocol_id=protocol_id,
        prompt_mode="teacher_chat_template_v1" if a.teacher else STUDENT_PROMPT_MODES[a.student_prompt_format],
        generation=generation_options,
        data_path=Path(a.data).resolve(),
        data_sha256=data_sha256,
        backend="vllm",
        weight_representation="base_plus_lora" if a.adapter else "base_weights",
        inference_dtype="bfloat16",
        model={
            "base_model": str(Path(a.model).resolve()),
            "checkpoint_path": str(Path(a.adapter).resolve()) if a.adapter else None,
            "checkpoint_sha256": adapter_sha256,
        },
    )
    config["runtime"] = {
        "batch_size": a.batch_size,
        "max_seqs": a.max_seqs,
        "max_num_batched_tokens": a.max_num_batched_tokens,
    }
    manifest = out / "generation-config.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != config:
            raise ValueError("Resume configuration mismatch")
    else:
        manifest.write_text(json.dumps(config, indent=2))
    rows = [json.loads(x) for x in Path(a.data).read_text().splitlines() if x.strip()]
    keys = [(r["domain"], str(r["id"])) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate benchmark ID")
    records = out / "generated.jsonl"
    done = set()
    if records.exists():
        with records.open("r+b") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    if f.read():
                        raise ValueError("Corruption before final record")
                    f.truncate(offset)
                    break
                k = (r["domain"], str(r["id"]))
                if k in done or k not in set(keys):
                    raise ValueError("Duplicate/foreign output")
                done.add(k)
    pending = [r for r in rows if (r["domain"], str(r["id"])) not in done]
    if not pending:
        return
    from transformers import AutoTokenizer
    from mopd.data.domains import student_prompt, teacher_content
    from mopd.data.templates import apply_chat_template_ids

    tok = AutoTokenizer.from_pretrained(a.model)
    work = collections.deque()
    for i, r in enumerate(pending):
        ids = (
            apply_chat_template_ids(
                tok,
                [{"role": "user", "content": teacher_content(r)}],
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if a.teacher
            else tok.encode(student_prompt(tok, r, a.student_prompt_format))
        )
        if len(ids) + a.max_new_tokens > a.max_model_len:
            raise ValueError("Prompt exceeds configured context: " + str(r["id"]))
        work.append((i, ids))
    lora_rank = None
    policy = None
    if a.adapter:
        r = json.loads((Path(a.adapter) / "adapter_config.json").read_text())["r"]
        lora_rank = next(size for size in (8, 16, 32, 64, 128, 256, 320, 512) if size >= r)
        policy = {"kind": "lora", "path": str(Path(a.adapter).resolve()), "id": adapter_sha256, "version": 0}
    engine = {
        "model": a.model,
        "max_model_len": a.max_model_len,
        "max_seqs": a.max_seqs,
        "max_num_batched_tokens": a.max_num_batched_tokens,
        "memory": 0.85,
        "prefix_caching": True,
        "max_lora_rank": lora_rank,
        "seed": 20261013,
    }
    session = {"max_tokens": a.max_new_tokens, "temperature": a.temperature, "top_p": a.top_p, "seed": 20261013}
    started = time.monotonic()
    written = 0
    per_gpu = collections.Counter()

    def stop(*_):
        raise SystemExit("terminated")

    signal.signal(signal.SIGTERM, stop)
    pool = EnginePool(a.device, engine, out, root / "cache/compile/gen")
    try:
        with records.open("a", buffering=1) as f:
            for key, m, device in pool.run(work, session, policy):
                row = pending[key]
                per_gpu[str(device)] += 1
                rec = {
                    "id": row["id"],
                    "domain": row["domain"],
                    "prompt": row["prompt"],
                    "response": m["text"],
                    "response_ids": m["token_ids"],
                    "tokens": len(m["token_ids"]),
                    "truncated": m["finish_reason"] == "length",
                    "finish_reason": m["finish_reason"],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                done.add((row["domain"], str(row["id"])))
                written += 1
                fields = dict(
                    done=len(done),
                    total=len(rows),
                    seconds_this_session=time.monotonic() - started,
                    backend="vllm",
                    per_gpu=dict(per_gpu),
                )
                progress.write(out / "generation-progress.json", **fields)  # every row: cheap, and this is the point
                if written % 20 == 0 or len(done) == len(rows):
                    os.fsync(f.fileno())  # durability; flush() above already makes rows visible to readers
                    print(json.dumps(fields), flush=True)
            os.fsync(f.fileno())
    finally:
        pool.close()
    if len(done) != len(rows):
        raise RuntimeError(f"generated {len(done)} of {len(rows)} rows")


if __name__ == "__main__":
    main()
