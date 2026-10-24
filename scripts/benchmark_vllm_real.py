"""Real prompt continuous-batching probe. Does not alter live training."""

import argparse, json, os, time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--batch-index", type=int, default=5)
    p.add_argument("--max-seqs", type=int, default=16)
    p.add_argument("--adapter")
    p.add_argument("--plan", default="/ssd/lzx/mopd/runs/20261014-dual-domain-batch32-mgpu-opd-b8/batch_plan.json")
    p.add_argument("--submission-batch", type=int, default=32)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.submission_batch < 1:
        raise ValueError("submission-batch must be positive")
    if a.device not in (2, 3):
        raise ValueError("Probe restricted to GPUs 2/3")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.device)
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from mopd.data.domains import student_prompt

    root = Path("/ssd/lzx/mopd")
    run = root / "runs/20261013-dual-domain-batch32-mgpu-opd"
    rows = json.loads(Path(a.plan).read_text())[a.batch_index]
    out = Path(a.out).resolve()
    if not out.is_relative_to(root):
        raise ValueError("Output outside project")
    out.mkdir(parents=True, exist_ok=False)
    engine = LLM(
        model=str(root / "models/Qwen3-4B-Base"),
        dtype="bfloat16",
        max_model_len=9216,
        generation_config="vllm",
        enable_prefix_caching=False,
        gpu_memory_utilization=0.85,
        max_num_seqs=a.max_seqs,
        max_num_batched_tokens=2048,
        enable_chunked_prefill=True,
        enable_lora=bool(a.adapter),
        max_lora_rank=16,
        seed=20261013,
    )
    tok = engine.get_tokenizer()
    prompts = [{"prompt_token_ids": tok.encode(student_prompt(tok, r))} for r in rows]
    params = SamplingParams(temperature=0, max_tokens=8192, stop_token_ids=[tok.eos_token_id])
    adapter = LoRARequest("probe", 1, a.adapter) if a.adapter else None
    engine.generate(prompts[:1], SamplingParams(temperature=0, max_tokens=8), lora_request=adapter, use_tqdm=False)
    start = time.perf_counter()
    outputs = []
    for offset in range(0, len(prompts), a.submission_batch):
        outputs.extend(
            engine.generate(prompts[offset : offset + a.submission_batch], params, lora_request=adapter, use_tqdm=False)
        )
    seconds = time.perf_counter() - start
    records = []
    for row, o in zip(rows, outputs):
        c = o.outputs[0]
        records.append(
            {
                "id": row["id"],
                "domain": row["domain"],
                "prompt": row["prompt"],
                "response": c.text,
                "tokens": len(c.token_ids),
                "finish_reason": c.finish_reason,
                "latency": o.metrics.finished_time - o.metrics.arrival_time if o.metrics else None,
            }
        )
    with (out / "cases.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    result = {
        "backend": "vllm",
        "device": a.device,
        "max_new_tokens": 8192,
        "max_seqs": a.max_seqs,
        "plan": a.plan,
        "submission_batch": a.submission_batch,
        "batch_index": a.batch_index,
        "adapter": a.adapter,
        "seconds": seconds,
        "tokens": sum(r["tokens"] for r in records),
        "longest": max(r["tokens"] for r in records),
        "truncated": sum(r["finish_reason"] == "length" for r in records),
        "sampling": "greedy; natural EOS; warmup excluded",
        "count": len(records),
    }
    result["tokens_per_second"] = result["tokens"] / seconds
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
