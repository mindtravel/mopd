"""Distinct prefixes, identical generated action tokens; no decode/re-encode."""
import torch
from mopd.data.templates import apply_chat_template_ids

def question(problem):
    return problem + "\nSolve step by step and put your final answer in \\boxed{}."

def student_inputs(tokenizer,problem,device):
    return tokenizer("Question: "+question(problem)+"\nAnswer:",return_tensors="pt").input_ids.to(device)

def teacher_inputs(tokenizer,problem,response_ids,device):
    prefix=apply_chat_template_ids(tokenizer,[{"role":"user","content":question(problem)}],
        add_generation_prompt=True,enable_thinking=False)
    prefix=torch.tensor([prefix],dtype=torch.long,device=device)
    return torch.cat([prefix,response_ids.to(device)],dim=1),prefix.shape[1]

def response_logps(logits,ids,start):
    if start<1 or start>=ids.shape[1]: raise ValueError("Empty response or prefix")
    # Chunk CE limits temporary FP32 allocations for the 152k vocabulary.
    pieces=[]
    for pos in range(start,ids.shape[1],64):
        end=min(pos+64,ids.shape[1])
        pieces.append(-torch.nn.functional.cross_entropy(
            logits[0,pos-1:end-1].float(),ids[0,pos:end],reduction="none"))
    return torch.cat(pieces)

def policy_loss(student_logps,teacher_logps):
    if student_logps.shape!=teacher_logps.shape: raise ValueError("Action mismatch")
    advantage=(teacher_logps.detach()-student_logps.detach()).clamp(-5,5)
    return -(advantage*student_logps).mean(),{
        "sampled_log_ratio":float((student_logps.detach()-teacher_logps.detach()).mean()),
        "clipped_fraction":float((advantage.abs()>=5).float().mean())}
