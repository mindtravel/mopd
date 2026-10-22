"""One independent vLLM GPU engine. Versioned LoRA, raw action IDs over stdio.

Each request either returns all answers in one reply (batch_generate) or, with "stream",
one reply per answer as soon as it finishes followed by a "done" summary (stream_generate).
Both run the same engine steps; streaming only changes when results leave the process.
"""
import argparse,contextlib,hashlib,json,os,sys,time,traceback
from pathlib import Path


def answer_item(row,output):
    o=output.outputs[0]
    return {"index":row["index"],"answer":list(o.token_ids),"finish_reason":o.finish_reason,
            "first_logprobs":({str(t):v.logprob for t,v in o.logprobs[0].items()} if o.logprobs else None)}


def batch_generate(engine,rows,params,adapter,emit,header):
    started=time.perf_counter()
    outputs=engine.generate([{"prompt_token_ids":r["prefix"]} for r in rows],params,lora_request=adapter,use_tqdm=False)
    result=[answer_item(r,o) for r,o in zip(rows,outputs)]
    emit({**header,"output":result,"seconds":time.perf_counter()-started,
          "count":len(result),"tokens":sum(len(r["answer"]) for r in result)})


def stream_generate(engine,rows,params,adapter,emit,header):
    """What LLM.generate does (add every request, then step the engine until none is left),
    except each finished sequence is emitted immediately. Request i gets id first_id + i."""
    started=time.perf_counter()
    prompts=[{"prompt_token_ids":r["prefix"]} for r in rows]
    first_id=engine.request_counter.counter
    engine._validate_and_add_requests(prompts=prompts,params=params,use_tqdm=False,
        lora_request=engine._get_modality_specific_lora_reqs(prompts,adapter),priority=None)
    if engine.llm_engine.get_num_unfinished_requests()!=len(rows):raise RuntimeError("Unexpected engine queue")
    sent=0;tokens=0
    while engine.llm_engine.has_unfinished_requests():
        for output in engine.llm_engine.step():
            if not output.finished:continue
            item=answer_item(rows[int(output.request_id)-first_id],output)
            sent+=1;tokens+=len(item["answer"])
            emit({**header,"partial":item})
    if sent!=len(rows):raise RuntimeError("Streamed fewer outputs than requests")
    emit({**header,"done":True,"seconds":time.perf_counter()-started,"count":sent,"tokens":tokens})


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
                header={"version":version,"adapter_sha256":digest,"device":a.device}
                run=stream_generate if request.get("stream") else batch_generate
                run(engine,request["inputs"],params,adapter,emit,header)
            except BaseException:
                emit({"error":traceback.format_exc()});raise
if __name__=="__main__":main()
