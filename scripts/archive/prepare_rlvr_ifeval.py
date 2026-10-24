"""Import pinned allenai/RLVR-IFeval and merge it with Tulu3 IF into one instruction pool.

RLVR-IFeval rows are Tulu 2 SFT prompts with one IFEval-family constraint added. Only the
user prompt is trained on; constraint and ground_truth are kept in metadata for verifiers.
"""
import collections,hashlib,json,random,re
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from mopd.data.pool import text_hash,validate_holdout,build_plan,load_pool
from mopd.protocol import student_prompt
ROOT=Path(__file__).resolve().parents[1]
SOURCE="allenai/RLVR-IFeval"
REVISION="47c03c73621c4aab2b824b7818681117d662770e"
MAX_PROMPT_TOKENS=1024  # vllm_worker max_model_len 9216 minus max_new_tokens 8192
SEED=20261014
def write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+"\n")
def norm(s):return re.sub(r"\W+"," ",s.lower()).strip()
def grams(s,n=8):
    w=norm(s).split();return {" ".join(w[i:i+n]) for i in range(len(w)-n+1)}
def main():
    root=ROOT/"data/instruction/rlvr_ifeval"
    parquet=root/"upstream/train-00000-of-00001.parquet"
    raw=pq.read_table(parquet).to_pylist()
    tok=AutoTokenizer.from_pretrained(str(ROOT/"models/Qwen3-4B-Base"),local_files_only=True)
    # IFEval + instruction dev + Tulu3 dev; Tulu3 train is excluded too so the merged pool has no duplicates.
    protected=load_pool(ROOT/"data/instruction/tulu3_if/protected.jsonl")
    tulu=load_pool(ROOT/"data/instruction/tulu3_if/train.jsonl")
    forbidden={text_hash(r["prompt"]) for r in protected}
    tulu_hashes={text_hash(r["prompt"]) for r in tulu}
    # 8-gram content overlap with IFEval, ignoring n-grams shared by >=5 RLVR prompts (constraint templates).
    df=collections.Counter(g for r in raw for g in grams(r["messages"][0]["content"]))
    template={g for g,c in df.items() if c>=5}
    ifeval_grams=set().union(*(grams(r["prompt"]) for r in load_pool(ROOT/"data/instruction/ifeval.jsonl")))-template
    seen=set();kept=[];excluded=collections.Counter()
    for i,r in enumerate(raw):
        if len(r["messages"])!=1 or r["messages"][0]["role"]!="user":excluded["not_single_user_turn"]+=1;continue
        prompt=r["messages"][0]["content"];h=text_hash(prompt)
        if h in forbidden:excluded["holdout_exact"]+=1;continue
        if h in tulu_hashes:excluded["tulu3_train_exact"]+=1;continue
        if h in seen:excluded["duplicate_prompt"]+=1;continue
        if grams(prompt)&ifeval_grams:excluded["ifeval_content_8gram"]+=1;continue
        row={"id":f"rlvr_ifeval-{i}","domain":"instruction","source":SOURCE,"prompt":prompt,
             "metadata":{"upstream_index":i,"constraint_type":r["constraint_type"],
                         "constraint":r["constraint"],"ground_truth":r["ground_truth"]}}
        if len(tok(student_prompt(tok,row,"plain_qa_v1")).input_ids)>MAX_PROMPT_TOKENS:
            excluded["prompt_over_max_tokens"]+=1;continue
        seen.add(h);kept.append(row)
    random.Random(SEED).shuffle(kept)
    dev=[dict(r,split="dev") for r in kept[:512]]
    train=[dict(r,split="train") for r in kept[512:]]
    write(root/"dev.jsonl",dev)
    write(root/"train.jsonl",train)
    validate_holdout(train,protected+dev)
    mixed=tulu+train
    build_plan(mixed,seed=SEED,steps=64,batch_size=16)
    validate_holdout(mixed,protected+dev)
    out=ROOT/"data/instruction/mixed_tulu3_rlvr_ifeval/train.jsonl"
    write(out,mixed)
    report={"source":SOURCE,"revision":REVISION,"license":"odc-by",
      "parquet_sha256":hashlib.sha256(parquet.read_bytes()).hexdigest(),
      "raw":len(raw),"train":len(train),"dev":len(dev),"excluded":dict(excluded),
      "constraint_type_counts":dict(collections.Counter(r["metadata"]["constraint_type"] for r in train)),
      "mixed_pool":{"path":str(out),"count":len(mixed),
                    "by_source":dict(collections.Counter(r["source"] for r in mixed))},
      "max_prompt_tokens":MAX_PROMPT_TOKENS,
      "leakage_check":"normalized exact vs IFEval/dev/Tulu3 dev/Tulu3 train; 8-gram content vs IFEval "
                      "excluding n-grams in >=5 RLVR prompts; semantic paraphrase not ruled out"}
    (root/"preparation_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False))
if __name__=="__main__":main()
