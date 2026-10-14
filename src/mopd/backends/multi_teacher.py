"""One student, independently frozen teachers; no optimizer owns teacher weights."""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from mopd.backends.hf import load_pair
def load_models(c):
    first=next(iter(c["domains"]))
    spec=c["domains"][first]
    tok,student,teacher=load_pair({**c,"teacher":spec["teacher"],"teacher_device":spec["device"]})
    teachers={first:(tok,teacher)}
    stok=AutoTokenizer.from_pretrained(c["student"],local_files_only=True)
    stok.padding_side="left"
    if stok.pad_token_id is None:stok.pad_token=stok.eos_token
    for domain,spec in c["domains"].items():
        if domain==first:continue
        tt=AutoTokenizer.from_pretrained(spec["teacher"],local_files_only=True)
        if tt.get_vocab()!=stok.get_vocab():raise ValueError("Vocabulary mismatch: "+domain)
        if tt.pad_token_id is None:tt.pad_token=tt.eos_token
        t=AutoModelForCausalLM.from_pretrained(spec["teacher"],torch_dtype=torch.bfloat16,attn_implementation="sdpa",local_files_only=True).to(spec["device"]).eval()
        t.requires_grad_(False);teachers[domain]=(tt,t)
    return stok,student,teachers
