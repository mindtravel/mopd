"""Check that MOPD rollouts do not depend on how many prompts share one vLLM request.

Sampled rollouts are random, so batch invariance is tested with greedy decoding: the same
student checkpoint generates the same prompts with requests of --batch prompts each
(max_num_seqs = --batch, like vllm_worker with generation_batch_size). Run once per setting,
on separate GPUs in parallel, then --compare the JSON outputs: per-prompt exact-match rate,
first divergence position, mean length, and throughput. Lossless when the cross-setting
agreement matches the same-setting repeat.

  mopd_infer/bin/python scripts/verify_rollout_batching.py --run runs/<mopd run> \
      --checkpoint checkpoint-0064 --steps 1,2 --batch 8 --device 5 --out tmp/rollout-b8.json
  python scripts/verify_rollout_batching.py --compare tmp/rollout-b8.json tmp/rollout-b16.json
"""
import argparse,json,os,time
from pathlib import Path
def generate(a):
    os.environ["CUDA_VISIBLE_DEVICES"]=str(a.device)
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer
    from mopd.protocol import student_prompt
    run=Path(a.run);c=json.loads((run/"config.json").read_text())
    tok=AutoTokenizer.from_pretrained(c["student"],local_files_only=True)
    plan=json.loads((run/"batch_plan.json").read_text())
    rows=[r for s in a.steps for r in plan[s-1]]
    prompts=[{"prompt_token_ids":tok.encode(student_prompt(tok,r,c["student_prompt_format"]))} for r in rows]
    engine=LLM(model=c["student"],dtype="bfloat16",max_model_len=9216,generation_config="vllm",enable_prefix_caching=False,
               gpu_memory_utilization=c.get("vllm_memory_utilization",.8),max_num_seqs=a.batch,max_num_batched_tokens=2048,
               enable_chunked_prefill=True,enable_lora=True,max_lora_rank=16,max_loras=1,max_cpu_loras=1)
    lora=LoRARequest("student",1,str(run/a.checkpoint))
    params=SamplingParams(temperature=0.,max_tokens=a.max_new_tokens,stop_token_ids=[tok.eos_token_id])
    outs=[];started=time.perf_counter()
    for i in range(0,len(prompts),a.batch):
        outs+=engine.generate(prompts[i:i+a.batch],params,lora_request=lora,use_tqdm=False)
    seconds=time.perf_counter()-started
    ids=[list(o.outputs[0].token_ids) for o in outs]
    Path(a.out).write_text(json.dumps({"batch":a.batch,"seconds":seconds,"tokens":sum(map(len,ids)),
        "row_ids":[r["id"] for r in rows],"outputs":ids}))
    print(json.dumps({"batch":a.batch,"seconds":round(seconds,1),"tok_per_s":round(sum(map(len,ids))/seconds)}))
def compare(paths):
    runs=[json.loads(Path(p).read_text()) for p in paths];ref=runs[0]
    for r in runs:
        same=[x==y for x,y in zip(r["outputs"],ref["outputs"])]
        div=[next((i for i,(u,v) in enumerate(zip(x,y)) if u!=v),min(len(x),len(y))) for x,y,s in zip(r["outputs"],ref["outputs"],same) if not s]
        print(json.dumps({"file":paths[runs.index(r)],"batch":r["batch"],"tok_per_s":round(r["tokens"]/r["seconds"]),
            "exact_match_vs_first":f"{sum(same)}/{len(same)}","median_first_divergence_token":sorted(div)[len(div)//2] if div else None,
            "mean_len":round(r["tokens"]/len(r["outputs"]),1)}))
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--compare",nargs="+")
    ap.add_argument("--run");ap.add_argument("--checkpoint");ap.add_argument("--steps",default="1,2")
    ap.add_argument("--batch",type=int);ap.add_argument("--device",type=int);ap.add_argument("--out")
    ap.add_argument("--max-new-tokens",type=int,default=2048);a=ap.parse_args()
    if a.compare:return compare(a.compare)
    a.steps=[int(x) for x in a.steps.split(",")];generate(a)
if __name__=="__main__":main()
