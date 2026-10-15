"""Training and evaluation protocol: what the student and teachers see, and what ends a generation.

Two model versions are only comparable when they used the same protocol, so every
decision about prompt text and EOS handling lives here and is recorded in the
model-version manifest. Nothing in this module imports the rest of mopd.
"""
from collections.abc import Mapping

# Prompt formats are part of the training protocol: an adapter must be rolled out and
# evaluated with the format it was trained on. Keys are config values; values are the
# prompt_mode recorded in benchmark generation manifests.
STUDENT_PROMPT_MODES = {
    "plain_qa_v1": "student_plain_qa_v1",
    "chat_template_v1": "student_checkpoint_chat_template_v1",
}

# plain_qa_v1 reproduces 20261014-dual-tulu3-from-base-vllm bit-exactly; verified by
# scripts/diagnostics/verify_student_prefix_paired.py.
PLAIN_QA_HEADERS = {"math": "Question: ", "instruction": "Instruction: "}


def apply_chat_template_text(tokenizer, messages, **kwargs):
    result = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    if not isinstance(result, str):
        raise TypeError("Chat template must return rendered text when tokenize=False")
    return result


def apply_chat_template_ids(tokenizer, messages, *, return_tensors=None, **kwargs):
    result = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors=return_tensors, **kwargs)
    if isinstance(result, Mapping):
        if "input_ids" not in result:
            raise ValueError("Chat template result has no input_ids")
        ids = result["input_ids"]
    elif hasattr(result, "input_ids"):
        ids = result.input_ids
    else:
        ids = result
    if return_tensors is not None:
        return ids
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        if len(ids) != 1:
            raise ValueError("Expected one chat-template sequence")
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def question(problem):
    return problem + "\nSolve step by step and put your final answer in \\boxed{}."


def teacher_content(row):
    """The user turn a teacher scores; identical for every teacher in a domain."""
    if row["domain"] == "math":
        return question(row["prompt"])
    if row["domain"] == "instruction":
        return row["prompt"]
    raise ValueError("Unknown domain: " + row["domain"])


def student_prompt(tokenizer, row, prompt_format):
    content = teacher_content(row)
    if prompt_format == "plain_qa_v1":
        return PLAIN_QA_HEADERS[row["domain"]] + content + "\nAnswer:"
    if prompt_format == "chat_template_v1":
        return apply_chat_template_text(
            tokenizer,
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            enable_thinking=False,
        )
    raise ValueError("Unknown student prompt format: " + repr(prompt_format)
                     + "; expected one of " + ", ".join(STUDENT_PROMPT_MODES))


def normalize_eos_token_ids(value):
    if value is None:
        return ()
    if hasattr(value, "tolist"):
        value = value.tolist()
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return tuple(dict.fromkeys(int(token) for token in values if token is not None))


def generation_eos_ids(model, tokenizer):
    generation = getattr(model, "generation_config", None)
    config = getattr(model, "config", None)
    for value in (getattr(generation, "eos_token_id", None),
                  getattr(config, "eos_token_id", None),
                  getattr(tokenizer, "eos_token_id", None)):
        ids = normalize_eos_token_ids(value)
        if ids:
            return ids
    raise ValueError("No EOS token IDs configured for generation")


def is_eos_token(token, eos):
    return int(token) in normalize_eos_token_ids(eos)


def trim_generated(tokens, eos):
    """Keep tokens up to and including the first EOS, so every answer ends where the model stopped."""
    values = tokens.tolist() if hasattr(tokens, "tolist") else list(tokens)
    eos_ids = set(normalize_eos_token_ids(eos))
    for index, token in enumerate(values):
        if token in eos_ids:
            return values[:index + 1]
    return values
