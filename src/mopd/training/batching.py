"""Masked batches with identical action IDs and sample-mean losses."""
import torch
from torch.utils.checkpoint import checkpoint
from mopd.protocol import (  # noqa: F401
    generation_eos_ids,
    is_eos_token,
    normalize_eos_token_ids,
    trim_generated,
)

def pack(sequences, starts, pad, device):
    width=max(map(len,sequences))
    ids=torch.full((len(sequences),width),pad,dtype=torch.long,device=device)
    attention=torch.zeros_like(ids);response=torch.zeros_like(ids,dtype=torch.bool)
    for i,(seq,start) in enumerate(zip(sequences,starts)):
        n=len(seq);ids[i,:n]=torch.as_tensor(seq,device=device)
        attention[i,:n]=1;response[i,start:n]=True
    return ids,attention,response

def logps(model, ids, attention, response):
    base=model.get_base_model() if hasattr(model,"get_base_model") else model
    positions=(attention.cumsum(-1)-1).clamp_min(0)
    hidden=base.model(input_ids=ids,attention_mask=attention,
        position_ids=positions,use_cache=False).last_hidden_state
    mask=response[:,1:] & attention[:,1:].bool()
    hs=hidden[:,:-1][mask];targets=ids[:,1:][mask]
    def head(h,y):
        return -torch.nn.functional.cross_entropy(base.lm_head(h).float(),y,reduction="none")
    values=[]
    for i in range(0,len(targets),128):
        h=hs[i:i+128];y=targets[i:i+128]
        values.append(checkpoint(head,h,y,use_reentrant=False) if torch.is_grad_enabled() else head(h,y))
    all_values=torch.cat(values)
    return list(all_values.split(mask.sum(1).tolist()))

def trim_generated(tokens,eos):
    values=tokens.tolist()
    eos_ids=set(normalize_eos_token_ids(eos))
    for index,token in enumerate(values):
        if token in eos_ids:
            return values[:index+1]
    return values
