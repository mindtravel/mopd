"""One formula for vLLM's max_model_len, so it cannot drift out of sync with max_new_tokens again.

Before this: generate_benchmark_vllm.py had max_model_len hardcoded to 16384 in two different
places (the engine constructor and a separate prompt-length check) while max_new_tokens was a
free CLI argument. Raising max_new_tokens past 16384 silently truncated every generation at the
engine level (no error) while the prompt-length check's own hardcoded 16384 raised on the first
row for an unrelated reason. Every caller now either omits --max-model-len (gets this default) or
passes one explicitly; there is no third place that independently hardcodes a context length.
"""

PROMPT_MARGIN = 4096  # headroom for the prompt + special tokens, on top of the generation budget
FLOOR = 16384  # unchanged default context for every existing protocol at max_new_tokens <= 12288


def default_max_model_len(max_new_tokens):
    return max(FLOOR, max_new_tokens + PROMPT_MARGIN)
