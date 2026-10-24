"""Generation batching is independent of scoring/backward microbatch size."""

import time
import torch
from mopd.data.domains import student_prompt
from mopd.training.batching import generation_eos_ids, trim_generated


def generate(
    model, tokenizer, rows, device, max_new_tokens, batch_size, *, prompt_format, sample=True, min_new_tokens=0
):
    model.eval()
    results = []
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    eos_ids = generation_eos_ids(model, tokenizer)
    for offset in range(0, len(rows), batch_size):
        items = rows[offset : offset + batch_size]
        inputs = tokenizer(
            [student_prompt(tokenizer, r, prompt_format) for r in items], padding=True, return_tensors="pt"
        ).to(device)
        kwargs = dict(
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=sample,
            eos_token_id=list(eos_ids),
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        if sample:
            kwargs.update(temperature=1.0, top_p=1.0, top_k=0)
        with torch.no_grad():
            generated = model.generate(**inputs, **kwargs)
        width = inputs.input_ids.shape[1]
        for i, row in enumerate(items):
            answer = trim_generated(generated[i, width:], eos_ids)
            prefix = inputs.input_ids[i][inputs.attention_mask[i].bool()].tolist()
            if not answer:
                raise ValueError("Empty generated response")
            results.append({"row": row, "prefix": prefix, "answer": answer})
        del generated, inputs
    torch.cuda.synchronize(device)
    return results, time.perf_counter() - started
