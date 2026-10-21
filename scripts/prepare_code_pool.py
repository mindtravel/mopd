"""Import one OpenCoder opc-sft-stage1 shard (filtered_infinity_instruct) as a code
SFT pool, decontaminated against HumanEval and MBPP the same way the instruction
pools are screened against IFEval: normalized exact-match plus 8-gram content overlap,
ignoring n-grams that recur across many rows (boilerplate, not real overlap).

Single-turn (instruction, output) pairs only; response length capped like the other
SFT pools. Reserves an independent 256-row dev split, same convention as
scripts/build_instruction_pilot.py.
"""
import collections,hashlib,json,random,re
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from mopd.data.pool import text_hash,validate_holdout
from mopd.protocol import student_prompt
ROOT=Path(__file__).resolve().parents[1]
SOURCE="OpenCoder-LLM/opc-sft-stage1:filtered_infinity_instruct"
REVISION="1bcab575f5e2d1c1fd6652720418524c27b3d58b"
MAX_PROMPT_TOKENS=1024
MAX_RESPONSE_TOKENS=1024
DEV_SIZE=256
SEED=20261021
def norm(s):return re.sub(r"\W+"," ",s.lower()).strip()
def grams(s,n=8):
    w=norm(s).split();return {" ".join(w[i:i+n]) for i in range(len(w)-n+1)}
def load_benchmark_prompts():
    prompts=[]
    he=pq.read_table(ROOT/"data/code/upstream/humaneval/test-00000-of-00001.parquet",columns=["prompt"]).to_pylist()
    prompts+=[r["prompt"].strip("'\"") for r in he]
    for split in ("prompt","test","train","validation"):
        mb=pq.read_table(ROOT/f"data/code/upstream/mbpp/sanitized-{split}-00000-of-00001.parquet",columns=["prompt"]).to_pylist()
        prompts+=[r["prompt"] for r in mb]
    return prompts
def main():
    out_root=ROOT/"data/sft/opencoder"
    tok=AutoTokenizer.from_pretrained(str(ROOT/"models/Qwen3-4B-Base"),local_files_only=True)
    raw=pq.read_table(ROOT/"data/code/upstream/opc_sft_stage1/filtered_infinity_instruct-00000-of-00004.parquet").to_pylist()
    benchmark_prompts=load_benchmark_prompts()
    # 8-gram content overlap vs HumanEval/MBPP, ignoring n-grams shared by >=5 training rows (boilerplate phrasing).
    df=collections.Counter(g for r in raw for g in grams(r["instruction"]))
    template={g for g,c in df.items() if c>=5}
    benchmark_grams=set().union(*(grams(p) for p in benchmark_prompts))-template
    seen=set();kept=[];excluded=collections.Counter()
    for i,r in enumerate(raw):
        prompt=(r.get("instruction") or "").strip();response=(r.get("output") or "").strip()
        if not prompt or not response:excluded["empty_field"]+=1;continue
        h=text_hash(prompt)
        if h in seen:excluded["duplicate_prompt"]+=1;continue
        if grams(prompt)&benchmark_grams:excluded["benchmark_content_8gram"]+=1;continue
        row={"id":f"opencoder_filtered_infinity-{i}","domain":"code","source":SOURCE,"prompt":prompt,
             "response":response,"split":"train","metadata":{"tag":r.get("tag")}}
        if len(tok(student_prompt(tok,row,"plain_qa_v1")).input_ids)>MAX_PROMPT_TOKENS:
            excluded["prompt_over_max_tokens"]+=1;continue
        if len(tok(response,add_special_tokens=False).input_ids)>MAX_RESPONSE_TOKENS:
            excluded["response_over_max_tokens"]+=1;continue
        seen.add(h);kept.append(row)
    random.Random(SEED).shuffle(kept)
    dev=[dict(r,split="dev") for r in kept[:DEV_SIZE]];train=kept[DEV_SIZE:]
    validate_holdout(train,dev)
    out_root.mkdir(parents=True,exist_ok=True)
    for name,rows in (("train",train),("dev",dev)):
        (out_root/f"{name}.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
    report={"source":SOURCE,"revision":REVISION,"license":"apache-2.0","raw":len(raw),
      "train":len(train),"dev":len(dev),"excluded":dict(excluded),
      "max_prompt_tokens":MAX_PROMPT_TOKENS,"max_response_tokens":MAX_RESPONSE_TOKENS,
      "decontamination":"HumanEval (164) + MBPP sanitized all splits (427) prompts; normalized exact-match "
                        "dedup plus 8-gram content overlap excluding n-grams shared by >=5 training rows; "
                        "semantic paraphrase and model pretraining contamination not ruled out",
      "note":"only one of 22 upstream opc-sft-stage1 shards downloaded; more shards (largescale_diverse_instruct, "
             "realuser_instruct) available if more code data is needed later",
      "train_sha256":hashlib.sha256((out_root/"train.jsonl").read_bytes()).hexdigest()}
    (out_root/"preparation_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False))
if __name__=="__main__":main()
