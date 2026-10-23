"""Validate rollout worker replies: policy version, policy identity (mopd.rollout.policy `id`),
and each sampled answer."""


def _check_policy(reply, version, digest):
    if reply.get("version") != version or reply.get("policy_id") != digest:
        raise ValueError("Stale policy or policy weights mismatch")


def validate_item(item, source, version, max_new_tokens, eos):
    """One sampled answer -> trajectory (the source input plus answer and policy version)."""
    answer = item["answer"]
    if not answer or len(answer) > max_new_tokens or any(type(t) is not int or t < 0 for t in answer):
        raise ValueError("Invalid sampled tokens")
    if eos in answer[:-1]:
        raise ValueError("Tokens after EOS")
    return {**source, "answer": answer, "student_version": version, "first_logprobs": item.get("first_logprobs")}


def validate_reply(reply, inputs, version, digest, max_new_tokens, eos):
    """A whole-batch reply: every input answered exactly once; returned in input order."""
    _check_policy(reply, version, digest)
    output = reply.get("output", [])
    expected = {r["index"]: r for r in inputs}
    indices = [r["index"] for r in output]
    if len(indices) != len(expected) or len(set(indices)) != len(indices) or set(indices) != set(expected):
        raise ValueError("Missing, duplicated, or unexpected rollout")
    found = {item["index"]: validate_item(item, expected[item["index"]], version, max_new_tokens, eos) for item in output}
    return [found[r["index"]] for r in inputs]


class StreamValidator:
    """Streamed replies for one request: each partial is validated on arrival, the final
    'done' message must account for every input exactly once."""

    def __init__(self, inputs, version, digest, max_new_tokens, eos):
        self.expected = {r["index"]: r for r in inputs}
        self.seen = set()
        self.args = (version, digest, max_new_tokens, eos)

    def partial(self, reply):
        version, digest, max_new_tokens, eos = self.args
        _check_policy(reply, version, digest)
        item = reply["partial"]
        index = item["index"]
        if index not in self.expected or index in self.seen:
            raise ValueError("Duplicated or unexpected rollout")
        self.seen.add(index)
        return validate_item(item, self.expected[index], version, max_new_tokens, eos)

    def done(self, reply):
        _check_policy(reply, self.args[0], self.args[1])
        if self.seen != set(self.expected) or reply.get("count") != len(self.expected):
            raise ValueError("Stream ended with missing rollouts")
        return {k: v for k, v in reply.items() if k not in ("done", "partial")}
