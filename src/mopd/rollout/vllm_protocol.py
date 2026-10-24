"""Validate each sampled answer coming back from a rollout engine (mopd.rollout.vllm_processes)."""


def validate_item(item, source, version, max_new_tokens, eos):
    """One sampled answer -> trajectory (the source input plus answer and policy version)."""
    answer = item["answer"]
    if not answer or len(answer) > max_new_tokens or any(type(t) is not int or t < 0 for t in answer):
        raise ValueError("Invalid sampled tokens")
    if eos in answer[:-1]:
        raise ValueError("Tokens after EOS")
    return {**source, "answer": answer, "student_version": version, "first_logprobs": item.get("first_logprobs")}
