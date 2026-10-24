"""One independent vLLM GPU engine running the student policy (LoRA adapter or full weights,
mopd.rollout.policy); raw action IDs over stdio.

Each request either returns all answers in one reply (batch_generate) or, with "stream",
one reply per answer as soon as it finishes followed by a "done" summary (stream_generate).
Both run the same engine steps; streaming only changes when results leave the process.
"""

import argparse, contextlib, hashlib, json, os, sys, time, traceback
from pathlib import Path


def answer_item(row, output):
    o = output.outputs[0]
    return {
        "index": row["index"],
        "answer": list(o.token_ids),
        "finish_reason": o.finish_reason,
        "first_logprobs": ({str(t): v.logprob for t, v in o.logprobs[0].items()} if o.logprobs else None),
    }


def batch_generate(engine, rows, params, adapter, emit, header):
    started = time.perf_counter()
    outputs = engine.generate(
        [{"prompt_token_ids": r["prefix"]} for r in rows], params, lora_request=adapter, use_tqdm=False
    )
    result = [answer_item(r, o) for r, o in zip(rows, outputs)]
    emit(
        {
            **header,
            "output": result,
            "seconds": time.perf_counter() - started,
            "count": len(result),
            "tokens": sum(len(r["answer"]) for r in result),
        }
    )


def stream_generate(engine, rows, params, adapter, emit, header):
    """What LLM.generate does (add every request, then step the engine until none is left),
    except each finished sequence is emitted immediately (mopd.rollout.engine_stream)."""
    from mopd.rollout.engine_stream import stream_requests

    started = time.perf_counter()
    prompts = [{"prompt_token_ids": r["prefix"]} for r in rows]
    sent = 0
    tokens = 0

    def on_finish(i, output):
        nonlocal sent, tokens
        item = answer_item(rows[i], output)
        sent += 1
        tokens += len(item["answer"])
        emit({**header, "partial": item})

    stream_requests(engine, prompts, params, adapter, on_finish)
    emit({**header, "done": True, "seconds": time.perf_counter() - started, "count": sent, "tokens": tokens})


def load_full_weights(worker, path):
    """Runs inside vLLM's worker (LLM.collective_rpc): replace every weight in place from one
    safetensors file (mopd.training.fsdp.write_weights)."""
    from safetensors import safe_open

    with safe_open(path, "pt", device="cpu") as f:
        worker.model_runner.model.load_weights((k, f.get_tensor(k)) for k in f.keys())


class Policy:
    """Which student weights the engine runs; switches only forward in version (mopd.rollout.policy)."""

    def __init__(self, engine, kind):
        self.engine, self.kind, self.version, self.id, self.adapter = engine, kind, None, None, None

    def use(self, p):
        from vllm.lora.request import LoRARequest

        if p["kind"] != self.kind:
            raise ValueError(f"worker runs {self.kind} students, got {p['kind']}")
        if self.version is not None and p["version"] < self.version:
            raise ValueError("Policy version went backwards")
        if p["version"] == self.version:
            if p["id"] != self.id:
                raise ValueError("Same version with different weights")
            return self.adapter
        if self.kind == "lora":
            path = Path(p["path"])
            if hashlib.sha256((path / "adapter_model.safetensors").read_bytes()).hexdigest() != p["id"]:
                raise ValueError("Adapter hash mismatch")
            if self.adapter is not None and not self.engine.llm_engine.remove_lora(self.adapter.lora_int_id):
                raise RuntimeError("Failed to remove previous policy")
            self.adapter = LoRARequest("student-" + str(p["version"]), p["version"] + 1, str(path))
            if not self.engine.llm_engine.add_lora(self.adapter):
                raise RuntimeError("LoRA load failed")
        else:
            self.engine.collective_rpc(load_full_weights, args=(p["path"],))
            self.engine.reset_prefix_cache()
        self.version, self.id = p["version"], p["id"]
        return self.adapter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--max-seqs", type=int, default=16)
    p.add_argument("--memory", type=float, default=0.85)
    p.add_argument("--student-kind", choices=("lora", "full"), default="lora")
    a = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.device.split(":")[-1]
    wire = sys.stdout

    def emit(value):
        wire.write("MOPD_JSON:" + json.dumps(value) + "\n")
        wire.flush()

    with contextlib.redirect_stdout(sys.stderr):
        from vllm import LLM, SamplingParams

        lora = a.student_kind == "lora"
        engine = LLM(
            model=a.model,
            dtype="bfloat16",
            max_model_len=9216,
            generation_config="vllm",
            enable_prefix_caching=False,
            gpu_memory_utilization=a.memory,
            max_num_seqs=a.max_seqs,
            max_num_batched_tokens=2048,
            enable_chunked_prefill=True,
            **({"enable_lora": True, "max_lora_rank": 16, "max_loras": 1, "max_cpu_loras": 1} if lora else {}),
        )
        eos = engine.get_tokenizer().eos_token_id
        emit({"ready": True, "device": a.device, "eos": eos})
        policy = Policy(engine, a.student_kind)
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request.get("stop"):
                    break
                adapter = policy.use(request["policy"])
                params = SamplingParams(
                    temperature=1.0 if request["sample"] else 0.0,
                    top_p=1.0,
                    top_k=-1,
                    max_tokens=request["max_new_tokens"],
                    min_tokens=request.get("min_new_tokens", 0),
                    stop_token_ids=[eos],
                    seed=request["seed"],
                    logprobs=request.get("debug_logprobs"),
                )
                header = {"version": policy.version, "policy_id": policy.id, "device": a.device}
                run = stream_generate if request.get("stream") else batch_generate
                run(engine, request["inputs"], params, adapter, emit, header)
            except BaseException:
                emit({"error": traceback.format_exc()})
                raise


if __name__ == "__main__":
    main()
