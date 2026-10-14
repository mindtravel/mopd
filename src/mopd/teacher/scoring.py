import torch
from mopd.data.templates import apply_chat_template_ids

def build_teacher_input(tokenizer, prompt, response):
    messages = [{"role":"user","content":prompt},{"role":"assistant","content":response}]
    return apply_chat_template_ids(tokenizer, messages,
        add_generation_prompt=False, enable_thinking=False, return_tensors="pt")

def score_response(teacher, tokenizer, prompt, response, student_response_ids,
                   teacher_device, target_device):
    teacher_ids = build_teacher_input(tokenizer, prompt, response).to(teacher_device)
    response_ids = tokenizer(response, add_special_tokens=False).input_ids
    student_ids = student_response_ids[0].tolist()
    starts = [i for i in range(teacher_ids.shape[1]-len(response_ids)+1)
              if teacher_ids[0,i:i+len(response_ids)].tolist() == response_ids]
    if not starts:
        raise ValueError("Could not locate response tokens in teacher template")
    start = starts[-1]; n = min(len(response_ids), len(student_ids))
    if response_ids[:n] != student_ids[:n]:
        raise ValueError("Student response changed during teacher re-tokenization")
    with torch.no_grad():
        logits = teacher(input_ids=teacher_ids, use_cache=False).logits
    return logits[:, start-1:start-1+n].to(target_device), teacher_ids.detach().cpu(), start, n
