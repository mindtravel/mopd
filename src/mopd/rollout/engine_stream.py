"""Drive a vLLM engine's request/step loop directly so each finished sequence is available the
moment it completes, instead of waiting for the whole submitted batch (what LLM.generate() does),
and so new requests can join the running continuous batch between steps.

Used by mopd.rollout.engine_worker (rows pulled from a coordinator as capacity frees up, for both
benchmark generation and rollouts) and tests/gpu/test_engine_stream.py: callers differ only in when
prompts arrive (`feed`) and what a finished output is used for (`on_finish`), so this is the one
place that touches vLLM's internal request/step API.
"""


def stream_feed(engine, params, adapter, feed, on_finish):
    """feed(unfinished) -> list of (key, prompt_dict) to add now, or None when no more will come.
    It is called before every engine step; it may block only when unfinished == 0 (nothing to
    step). on_finish(key, output) runs once per finished request, in finish order.

    Returns counters for mopd.profiling.model.infer_utilization: engine steps, generated and
    prefilled tokens, and the attention context read while decoding (prompt p, answer g:
    p*g + g*(g+1)/2 per request). Integer additions per step: no measurable cost."""
    keys, unfinished, closed, added = {}, 0, False, 0
    counters = {"steps": 0, "generated": 0, "prefilled": 0, "context_tokens": 0}
    while True:
        if not closed:
            new = feed(unfinished)
            if new is None:
                closed = True
            elif new:
                # LLMEngine.add_request with our own ids: the engine-level API, stable across vLLM
                # 0.10 -> 0.30 (LLM's private _validate_and_add_requests is gone in 0.30)
                for k, prompt in new:
                    rid = f"s{added}"
                    keys[engine.llm_engine.add_request(rid, prompt, params, lora_request=adapter) or rid] = k
                    added += 1
                unfinished += len(new)
        if unfinished == 0:
            if closed:
                return counters
            continue
        if not engine.llm_engine.has_unfinished_requests():
            raise RuntimeError(f"engine lost {unfinished} request(s)")
        counters["steps"] += 1
        for output in engine.llm_engine.step():
            if output.finished:
                p, g = len(output.prompt_token_ids or ()), len(output.outputs[0].token_ids)
                counters["generated"] += g
                counters["prefilled"] += p
                counters["context_tokens"] += p * g + g * (g + 1) // 2
                on_finish(keys.pop(output.request_id), output)
                unfinished -= 1


def stream_requests(engine, prompts, params, adapter, on_finish):
    """Submit every prompt at once; on_finish(request_index, output) in finish order."""
    batch = [list(enumerate(prompts))]
    return stream_feed(engine, params, adapter, lambda unfinished: batch.pop() if batch else None, on_finish)
