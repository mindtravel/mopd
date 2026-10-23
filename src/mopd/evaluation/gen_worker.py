"""One GPU's resident vLLM engine for benchmark generation, pulling rows from the coordinator
(scripts/generate_benchmark_vllm.py) whenever it has spare capacity.

It asks for a few more rows whenever its scheduler's waiting queue is empty; they join the running
continuous batch at the next step. The coordinator caps every GPU at a fair share of the unfinished
rows and answers an empty list beyond it, after which this worker waits for one of its rows to
finish before asking again. Rows nobody holds stay in the coordinator's queue, where any GPU can
take them -- so the last rows of a benchmark spread over every GPU.

The engine core runs in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0, set by the coordinator):
the scheduler's queue counts are readable directly, and killing this process leaves no orphaned
EngineCore holding the GPU.

Wire (JSON lines, worker output prefixed with TAG):
  worker -> coordinator: {"ready"} {"want": k} {"key", "text", "token_ids", "finish_reason"} {"error"}
  coordinator -> worker: {"rows": [[key, prompt_token_ids], ...]} (may be empty) or {"eof": true};
  one reply per want.
"""
import contextlib, json, queue, sys, threading, traceback

TAG = "GEN_JSON:"
ASK = 16


def main():
    c = json.loads(sys.argv[1])
    wire = sys.stdout

    def emit(value):
        wire.write(TAG + json.dumps(value) + "\n"); wire.flush()

    inbox = queue.Queue()

    def read():
        for line in sys.stdin:
            inbox.put(json.loads(line))
        inbox.put({"lost": True})  # coordinator gone

    threading.Thread(target=read, daemon=True).start()
    with contextlib.redirect_stdout(sys.stderr):
        try:
            from vllm import LLM, SamplingParams
            from vllm.lora.request import LoRARequest
            from mopd.rollout.engine_stream import stream_feed
            engine = LLM(model=c["model"], dtype="bfloat16", max_model_len=c["max_model_len"],
                gpu_memory_utilization=.85, max_num_seqs=c["max_seqs"], max_num_batched_tokens=c["max_num_batched_tokens"],
                enable_chunked_prefill=True, enable_prefix_caching=True, generation_config="vllm",
                enable_lora=bool(c["adapter"]), max_lora_rank=16, seed=20261013)
            scheduler = engine.llm_engine.engine_core.engine_core.scheduler
            eos = engine.get_tokenizer().eos_token_id
            sampled = c["temperature"] > 0
            params = SamplingParams(temperature=c["temperature"], top_p=c["top_p"] if sampled else 1.0,
                seed=20261013 if sampled else None, max_tokens=c["max_new_tokens"], stop_token_ids=[eos])
            adapter = LoRARequest("evaluated-student", 1, c["adapter"]) if c["adapter"] else None
            emit({"ready": True, "eos": eos})
            asked = False
            full = False  # last reply was empty: this GPU holds its fair share; ask again once a row finishes

            def feed(unfinished):
                nonlocal asked, full
                if not asked and not full and scheduler.get_request_counts()[1] == 0:
                    emit({"want": ASK}); asked = True
                try:
                    msg = inbox.get(block=unfinished == 0)
                except queue.Empty:
                    return []
                if msg.get("lost"):
                    raise SystemExit("coordinator closed the pipe")
                asked = False
                if msg.get("eof"):
                    return None
                full = not msg["rows"]
                return [(k, {"prompt_token_ids": ids}) for k, ids in msg["rows"]]

            def on_finish(key, output):
                nonlocal full
                full = False
                o = output.outputs[0]
                emit({"key": key, "text": o.text, "token_ids": list(o.token_ids), "finish_reason": o.finish_reason})

            stream_feed(engine, params, adapter, feed, on_finish)
        except BaseException:
            emit({"error": traceback.format_exc()})
            raise


if __name__ == "__main__":
    main()
