"""One GPU's resident vLLM engine, the only one in the project: benchmark generation and on-policy
rollouts both drive it through mopd.rollout.engine_pool.EnginePool.

Work comes in sessions. A session names the sampling parameters and the weights (base, a LoRA
adapter, or the trainer's shared full-weight buffer -- mopd.rollout.policy); within it the worker asks
for rows whenever its scheduler's waiting queue is empty, and they join the running continuous batch
at the next step, so no GPU idles while another still holds a backlog. After the coordinator's eof and
the last row, the worker reports idle and waits for the next session (one engine per run, reused every
training step).

The engine core runs in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0, set by EnginePool): the
scheduler's queue counts are readable directly and killing the process leaves no orphaned EngineCore.

Wire (JSON lines; worker output prefixed with TAG):
  -> worker: {"session": {...sampling..., "session": n}, "policy": descriptor | null}  starts a session
             {"rows": [[key, prompt_token_ids], ...]}  (may be empty)  answers a want
             {"eof": true}  no more rows this session        {"stop": true}  exit
  worker ->: {"ready", "eos"}  {"want": k}  {"key", "text", "token_ids", "finish_reason", "first_logprobs"}
             {"idle": true, "session": n, "policy_id", "load_seconds"}  {"error": traceback}
"""

import contextlib, hashlib, json, queue, sys, threading, time, traceback
from pathlib import Path

TAG = "ENGINE_JSON:"
ASK = 16


def load_full_weights(worker, path):
    """Runs inside vLLM's worker (LLM.collective_rpc): replace every weight in place from the
    trainer's shared bf16 buffer (mopd.training.fsdp.SharedWeights), mapped without copying."""
    from mopd.training.fsdp import read_shared_weights

    worker.model_runner.model.load_weights(read_shared_weights(path))


class Policy:
    """Which weights the engine runs; versions only move forward."""

    def __init__(self, engine):
        self.engine, self.version, self.id, self.adapter, self.load_seconds = engine, None, None, None, 0.0

    def use(self, p):
        from vllm.lora.request import LoRARequest

        self.load_seconds = 0.0
        if p is None:
            return None
        if self.version is not None and p["version"] < self.version:
            raise ValueError("Policy version went backwards")
        if p["version"] == self.version:
            if p["id"] != self.id:
                raise ValueError("Same version with different weights")
            return self.adapter
        mark = time.perf_counter()
        if p["kind"] == "lora":
            path = Path(p["path"])
            if hashlib.sha256((path / "adapter_model.safetensors").read_bytes()).hexdigest() != p["id"]:
                raise ValueError("Adapter hash mismatch")
            if self.adapter is not None and not self.engine.llm_engine.remove_lora(self.adapter.lora_int_id):
                raise RuntimeError("Failed to remove previous policy")
            self.adapter = LoRARequest(f"policy-{p['version']}", p["version"] + 1, str(path))
            if not self.engine.llm_engine.add_lora(self.adapter):
                raise RuntimeError("LoRA load failed")
        elif p["kind"] == "full":
            self.engine.collective_rpc(load_full_weights, args=(p["path"],))
            self.engine.reset_prefix_cache()
        else:
            raise ValueError(f"unknown policy kind {p['kind']!r}")
        self.version, self.id = p["version"], p["id"]
        self.load_seconds = time.perf_counter() - mark
        return self.adapter


def main():
    c = json.loads(sys.argv[1])
    wire = sys.stdout

    def emit(value):
        wire.write(TAG + json.dumps(value) + "\n")
        wire.flush()

    inbox = queue.Queue()

    def read():
        for line in sys.stdin:
            inbox.put(json.loads(line))
        inbox.put({"stop": True, "lost": True})  # coordinator gone

    threading.Thread(target=read, daemon=True).start()
    with contextlib.redirect_stdout(sys.stderr):
        try:
            from vllm import LLM, SamplingParams
            from mopd.rollout.engine_stream import stream_feed

            lora = c.get("max_lora_rank")
            engine = LLM(
                model=c["model"],
                dtype="bfloat16",
                max_model_len=c["max_model_len"],
                gpu_memory_utilization=c.get("memory", 0.85),
                max_num_seqs=c["max_seqs"],
                max_num_batched_tokens=c["max_num_batched_tokens"],
                enable_chunked_prefill=True,
                enable_prefix_caching=c.get("prefix_caching", True),
                generation_config="vllm",
                seed=c.get("seed", 20261013),
                **({"enable_lora": True, "max_lora_rank": lora, "max_loras": 1, "max_cpu_loras": 1} if lora else {}),
            )
            scheduler = engine.llm_engine.engine_core.engine_core.scheduler
            eos = engine.get_tokenizer().eos_token_id
            policy = Policy(engine)
            emit({"ready": True, "eos": eos})
            while True:
                msg = inbox.get()
                if msg.get("stop"):
                    if msg.get("lost"):
                        raise SystemExit("coordinator closed the pipe")
                    return
                s = msg["session"]
                adapter = policy.use(msg.get("policy"))
                sampled = s.get("temperature", 0.0) > 0
                params = SamplingParams(
                    temperature=s.get("temperature", 0.0),
                    top_p=s.get("top_p", 1.0) if sampled else 1.0,
                    top_k=-1,
                    seed=s.get("seed") if sampled else None,
                    max_tokens=s["max_tokens"],
                    min_tokens=s.get("min_tokens", 0),
                    stop_token_ids=[eos],
                    logprobs=s.get("logprobs"),
                )
                asked, full = False, False  # full: last reply was empty -- ask again once a row finishes

                def feed(unfinished):
                    nonlocal asked, full
                    if not asked and not full and scheduler.get_request_counts()[1] == 0:
                        emit({"want": ASK})
                        asked = True
                    try:
                        m = inbox.get(block=unfinished == 0)
                    except queue.Empty:
                        return []
                    if m.get("stop"):
                        raise SystemExit("stopped mid-session")
                    asked = False
                    if m.get("eof"):
                        return None
                    full = not m["rows"]
                    return [(k, {"prompt_token_ids": ids}) for k, ids in m["rows"]]

                def on_finish(key, output):
                    nonlocal full
                    full = False
                    o = output.outputs[0]
                    first = {str(t): v.logprob for t, v in o.logprobs[0].items()} if o.logprobs else None
                    emit(
                        {
                            "key": key,
                            "text": o.text,
                            "token_ids": list(o.token_ids),
                            "finish_reason": o.finish_reason,
                            "first_logprobs": first,
                        }
                    )

                stream_feed(engine, params, adapter, feed, on_finish)
                emit(
                    {
                        "idle": True,
                        "session": s.get("session"),
                        "policy_id": policy.id,
                        "load_seconds": policy.load_seconds,
                    }
                )
        except BaseException:
            emit({"error": traceback.format_exc()})
            raise


if __name__ == "__main__":
    main()
