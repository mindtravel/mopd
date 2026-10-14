"""One independent vLLM GPU engine. Versioned LoRA, raw action IDs over stdio."""
import argparse,contextlib,hashlib,json,os,sys,time,traceback
from pathlib import Path
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True);p.add_argument("--device",required=True)
    p.add_argument("--max-seqs",type=int,default=16);p.add_argument("--memory",type=float,default=.85)
    a=p.parse_args()
    device=int(a.device.split(":")[-1])
    if device not in (2,3,5,6,7):raise ValueError("Unapproved rollout GPU")
    os.environ["CUDA_VISIBLE_DEVICES"]=str(device)
    wire=sys.stdout
    def emit(value):
        wire.write("MOPD_JSON:"+json.dumps(value)+"\n");wire.flush()
    with contextlib.redirect_stdout(sys.stderr):
        from vllm import LLM,SamplingParams
        from vllm.lora.request import LoRARequest
        engine=LLM(model=a.model,dtype="bfloat16",max_model_len=9216,
            generation_config="vllm",enable_prefix_caching=False,
            gpu_memory_utilization=a.memory,max_num_seqs=a.max_seqs,
            max_num_batched_tokens=2048,enable_chunked_prefill=True,
            enable_lora=True,max_lora_rank=16,max_loras=1,max_cpu_loras=1)
        eos=engine.get_tokenizer().eos_token_id
        emit({"ready":True,"device":a.device,"eos":eos})
        active=None;active_digest=None
        for line in sys.stdin:
            try:
                request=json.loads(line)
                if request.get("stop"):break
                path=Path(request["adapter"])
                digest=hashlib.sha256((path/"adapter_model.safetensors").read_bytes()).hexdigest()
                if digest!=request["adapter_sha256"]:raise ValueError("Adapter hash mismatch")
                version=request["version"];adapter_id=version+1
                if active==adapter_id and active_digest!=digest:raise ValueError("Same version with different weights")
                if active is not None and adapter_id<active:raise ValueError("Policy version went backwards")
                if active!=adapter_id:
                    if active is not None and not engine.llm_engine.remove_lora(active):
                        raise RuntimeError("Failed to remove previous policy")
                    adapter=LoRARequest("student-"+str(version),adapter_id,str(path))
                    if not engine.llm_engine.add_lora(adapter):raise RuntimeError("LoRA load failed")
                    active=adapter_id;active_digest=digest
                else:adapter=LoRARequest("student-"+str(version),adapter_id,str(path))
                params=SamplingParams(temperature=1. if request["sample"] else 0.,
                    top_p=1.,top_k=-1,max_tokens=request["max_new_tokens"],
                    min_tokens=request.get("min_new_tokens",0),
                    stop_token_ids=[eos],seed=request["seed"],logprobs=request.get("debug_logprobs"))
                rows=request["inputs"]
                prompts=[{"prompt_token_ids":r["prefix"]} for r in rows]
                started=time.perf_counter()
                outputs=engine.generate(prompts,params,lora_request=adapter,use_tqdm=False)
                result=[{"index":r["index"],"answer":list(o.outputs[0].token_ids),
                    "finish_reason":o.outputs[0].finish_reason,
                    "first_logprobs":({str(t):v.logprob for t,v in o.outputs[0].logprobs[0].items()} if o.outputs[0].logprobs else None)} for r,o in zip(rows,outputs)]
                emit({"version":version,"adapter_sha256":digest,"output":result,
                    "seconds":time.perf_counter()-started,"device":a.device,
                    "count":len(result),"tokens":sum(len(r["answer"]) for r in result)})
            except BaseException:
                emit({"error":traceback.format_exc()});raise
if __name__=="__main__":main()
