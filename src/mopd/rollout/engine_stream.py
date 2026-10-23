"""Drive a vLLM engine's request/step loop directly so each finished sequence is available the
moment it completes, instead of waiting for the whole submitted batch (what LLM.generate() does).
Every prompt is still submitted at once, so vLLM's continuous batching schedules against the full
pool for throughput exactly as LLM.generate() would -- only how results leave the engine changes.

Shared by mopd.rollout.vllm_worker (MOPD's training rollout, token-id trajectories) and
scripts/generate_benchmark_vllm.py (benchmark generation, text + prompt records): they differ only
in what a finished output is used for, expressed as the `on_finish` callback, so this is the one
place that touches vLLM's internal request/step API.
"""


def stream_requests(engine, prompts, params, adapter, on_finish):
    """prompts: vLLM's own generate() prompt dicts (e.g. [{"prompt_token_ids": [...]}, ...]).
    on_finish(request_index, output) runs once per finished request, in finish order (not
    necessarily submission order); request_index is that prompt's position in `prompts`."""
    first_id = engine.request_counter.counter
    engine._validate_and_add_requests(prompts=prompts, params=params, use_tqdm=False,
        lora_request=engine._get_modality_specific_lora_reqs(prompts, adapter), priority=None)
    if engine.llm_engine.get_num_unfinished_requests() != len(prompts):
        raise RuntimeError("Unexpected engine queue")
    finished = 0
    while engine.llm_engine.has_unfinished_requests():
        for output in engine.llm_engine.step():
            if not output.finished:
                continue
            on_finish(int(output.request_id) - first_id, output)
            finished += 1
    if finished != len(prompts):
        raise RuntimeError("Streamed fewer outputs than requests")
