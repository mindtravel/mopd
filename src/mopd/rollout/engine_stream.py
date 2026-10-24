"""Drive a vLLM engine's request/step loop directly so each finished sequence is available the
moment it completes, instead of waiting for the whole submitted batch (what LLM.generate() does),
and so new requests can join the running continuous batch between steps.

Shared by mopd.rollout.vllm_worker (MOPD's training rollout, everything submitted at once) and
mopd.evaluation.gen_worker (benchmark generation, rows pulled from a coordinator as capacity frees
up): they differ only in when prompts arrive (`feed`) and what a finished output is used for
(`on_finish`), so this is the one place that touches vLLM's internal request/step API.
"""


def stream_feed(engine, params, adapter, feed, on_finish):
    """feed(unfinished) -> list of (key, prompt_dict) to add now, or None when no more will come.
    It is called before every engine step; it may block only when unfinished == 0 (nothing to
    step). on_finish(key, output) runs once per finished request, in finish order."""
    keys, unfinished, closed = {}, 0, False
    while True:
        if not closed:
            new = feed(unfinished)
            if new is None:
                closed = True
            elif new:
                first = engine.request_counter.counter
                prompts = [p for _, p in new]
                engine._validate_and_add_requests(
                    prompts=prompts,
                    params=params,
                    use_tqdm=False,
                    lora_request=engine._get_modality_specific_lora_reqs(prompts, adapter),
                    priority=None,
                )
                keys.update({str(first + i): k for i, (k, _) in enumerate(new)})
                unfinished += len(new)
        if unfinished == 0:
            if closed:
                return
            continue
        if not engine.llm_engine.has_unfinished_requests():
            raise RuntimeError(f"engine lost {unfinished} request(s)")
        for output in engine.llm_engine.step():
            if output.finished:
                on_finish(keys.pop(output.request_id), output)
                unfinished -= 1


def stream_requests(engine, prompts, params, adapter, on_finish):
    """Submit every prompt at once; on_finish(request_index, output) in finish order."""
    batch = [list(enumerate(prompts))]
    stream_feed(engine, params, adapter, lambda unfinished: batch.pop() if batch else None, on_finish)
