"""Validate rollout identity without decoding or retokenizing sampled actions."""
def validate_reply(reply, inputs, version, digest, max_new_tokens, eos):
    if reply.get("version") != version or reply.get("adapter_sha256") != digest:
        raise ValueError("Stale policy or adapter file mismatch")
    output=reply.get("output",[])
    expected={r["index"]:r for r in inputs}
    indices=[r["index"] for r in output]
    if len(indices)!=len(expected) or len(set(indices))!=len(indices) or set(indices)!=set(expected):
        raise ValueError("Missing, duplicated, or unexpected rollout")
    found={}
    for item in output:
        answer=item["answer"]
        if not answer or len(answer)>max_new_tokens or any(type(t) is not int or t<0 for t in answer):
            raise ValueError("Invalid sampled tokens")
        if eos in answer[:-1]:
            raise ValueError("Tokens after EOS")
        source=expected[item["index"]]
        found[item["index"]]={**source,"answer":answer,"student_version":version,
            "first_logprobs":item.get("first_logprobs")}
    return [found[r["index"]] for r in inputs]
