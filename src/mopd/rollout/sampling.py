import torch
from mopd.training.batching import generation_eos_ids
from mopd.protocol import apply_chat_template_text


def build_student_prompt(tokenizer, prompt):
    return apply_chat_template_text(
        tokenizer,
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generate(student, tokenizer, prompt, device, max_new_tokens, seed):
    torch.manual_seed(seed)
    text = build_student_prompt(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    student.eval()
    eos_ids = generation_eos_ids(student, tokenizer)
    with torch.no_grad():
        ids = student.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=list(eos_ids),
            use_cache=True,
        )
    n = inputs.input_ids.shape[1]
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[:, n:] = True
    return ids, mask, tokenizer.decode(ids[0, n:], skip_special_tokens=False), text
