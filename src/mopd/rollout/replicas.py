"""Synchronous multi-GPU rollout replicas: fresh LoRA before every generation."""
from concurrent.futures import ThreadPoolExecutor
import copy,time
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import get_peft_model
from mopd.rollout.hf_batch import generate

def assign_shards(rows,workers):
    if workers<1:raise ValueError("No workers")
    return [rows[i::workers] for i in range(workers)]

def sync_lora(source,target):
    src={n:p for n,p in source.named_parameters() if "lora_" in n}
    dst={n:p for n,p in target.named_parameters() if "lora_" in n}
    if not src or set(src)!=set(dst):raise ValueError("LoRA structure mismatch")
    with torch.no_grad():
        for name,p in src.items():
            if dst[name].shape!=p.shape:raise ValueError("LoRA shape mismatch")
            dst[name].copy_(p.detach().to(dst[name].device))
        for name,p in src.items():
            if not torch.equal(dst[name],p.detach().to(dst[name].device)):raise ValueError("LoRA sync mismatch")
    for device in {p.device for p in dst.values()}:
        if device.type=="cuda":torch.cuda.synchronize(device)

class RolloutReplicas:
    def __init__(self,c,student):
        self.workers=[];self.max_new=c["max_new_tokens"];self.batch=c["generation_batch_size"];self.prompt_format=c["student_prompt_format"]
        self.versions={}
        for device in c["rollout_devices"]:
            tok=AutoTokenizer.from_pretrained(c["student"],local_files_only=True);tok.padding_side="left"
            if tok.pad_token_id is None:tok.pad_token=tok.eos_token
            base=AutoModelForCausalLM.from_pretrained(c["student"],torch_dtype=torch.bfloat16,attn_implementation="sdpa",local_files_only=True).to(device)
            replica=get_peft_model(base,copy.deepcopy(student.peft_config["default"]))
            replica.requires_grad_(False);replica.eval()
            self.workers.append((device,tok,replica))
    def generate(self,student,rows,version,*,sample=True,min_new_tokens=0):
        started=time.perf_counter()
        for device,_,model in self.workers:
            sync_lora(student,model);self.versions[device]=version
        sync_seconds=time.perf_counter()-started
        shards=assign_shards(list(enumerate(rows)),len(self.workers))
        def run(worker,shard):
            device,tok,model=worker
            if not shard:return [],{}
            with torch.cuda.device(device):
                torch.cuda.reset_peak_memory_stats(device)
                output,seconds=generate(model,tok,[r for _,r in shard],device,self.max_new,self.batch,prompt_format=self.prompt_format,sample=sample,min_new_tokens=min_new_tokens)
                for item,(index,_) in zip(output,shard):
                    item["index"]=index;item["rollout_device"]=device;item["student_version"]=version
                return output,{"device":device,"version":version,"count":len(output),"seconds":seconds,
                    "tokens":sum(len(r["answer"]) for r in output),"peak_gib":torch.cuda.max_memory_allocated(device)/2**30}
        started=time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(self.workers)) as pool:
            futures=[pool.submit(run,w,s) for w,s in zip(self.workers,shards)]
            results=[f.result() for f in futures]
        seconds=time.perf_counter()-started
        output=sorted([r for items,_ in results for r in items],key=lambda r:r["index"])
        if len(output)!=len(rows) or any(r["student_version"]!=version for r in output):raise ValueError("Stale or missing rollout")
        return output,seconds,{"sync_seconds":sync_seconds,"workers":[stats for _,stats in results]}
