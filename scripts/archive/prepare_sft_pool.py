"""Import pinned HuggingFaceTB/smoltalk smol-constraints with assistant answers kept, for real SFT.

Same upstream parquet, same filtering rules as scripts/prepare_smol_constraints.py (that script
builds the OPD prompt pool and discards the reference answer because an OPD teacher must never
see it). Here the answer is the training target, so it is kept instead of dropped. Reuses the
already-downloaded parquet; no new network access.
"""
import collections,hashlib,json,re
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from mopd.data.pool import text_hash,validate_holdout,load_pool
from mopd.protocol import student_prompt
ROOT=Path(__file__).resolve().parents[1]
SOURCE="HuggingFaceTB/smoltalk:smol-constraints"
REVISION="5feaf2fd3ffca7c237fc38d1861bc30365d48ffa"
MAX_PROMPT_TOKENS=1024  # same bound as the OPD pool; keeps prompts short relative to responses.
MAX_RESPONSE_TOKENS=1024
def write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+"\n")
def norm(s):return re.sub(r"\W+"," ",s.lower()).strip()
def grams(s,n=8):
    w=norm(s).split();return {" ".join(w[i:i+n]) for i in range(len(w)-n+1)}
def answer_in_prompt(prompt,answer,n=12):
    return bool(grams(answer,n)&grams(prompt,n))
def load_split(root,split):
    return pq.read_table(root/f"upstream/data/smol-constraints/{split}-00000-of-00001.parquet").to_pylist()
def main():
    upstream_root=ROOT/"data/instruction/smol_constraints"
    out_root=ROOT/"data/sft/smol_constraints"
    tok=AutoTokenizer.from_pretrained(str(ROOT/"models/Qwen3-4B-Base"),local_files_only=True)
    protected=load_pool(ROOT/"data/instruction/tulu3_if/protected.jsonl")
    tulu=load_pool(ROOT/"data/instruction/tulu3_if/train.jsonl")
    forbidden={text_hash(r["prompt"]) for r in protected}
    tulu_hashes={text_hash(r["prompt"]) for r in tulu}
    raw={s:load_split(upstream_root,s) for s in ("train","test")}
    # 8-gram content overlap with IFEval, ignoring n-grams shared by >=5 prompts (constraint templates).
    df=collections.Counter(g for rows in raw.values() for r in rows for g in grams(r["messages"][0]["content"]))
    template={g for g,c in df.items() if c>=5}
    ifeval_grams=set().union(*(grams(r["prompt"]) for r in load_pool(ROOT/"data/instruction/ifeval.jsonl")))-template
    seen=set();out={};excluded={}
    for split in ("test","train"):  # test first so train is deduplicated against dev
        kept=[];ex=collections.Counter()
        for i,r in enumerate(raw[split]):
            m=r["messages"]
            if len(m)!=2 or m[0]["role"]!="user" or m[1]["role"]!="assistant":ex["not_single_turn"]+=1;continue
            prompt=m[0]["content"];response=m[1]["content"];h=text_hash(prompt)
            if h in forbidden:ex["holdout_exact"]+=1;continue
            if h in tulu_hashes:ex["tulu3_train_exact"]+=1;continue
            if h in seen:ex["duplicate_prompt"]+=1;continue
            if answer_in_prompt(prompt,response):ex["reference_answer_in_prompt"]+=1;continue
            if grams(prompt)&ifeval_grams:ex["ifeval_content_8gram"]+=1;continue
            row={"id":f"smol_constraints_sft-{split}-{i}","domain":"instruction","source":SOURCE,"prompt":prompt,
                 "response":response,"split":"dev" if split=="test" else "train",
                 "metadata":{"upstream_split":split,"upstream_index":i}}
            if len(tok(student_prompt(tok,row,"plain_qa_v1")).input_ids)>MAX_PROMPT_TOKENS:
                ex["prompt_over_max_tokens"]+=1;continue
            if len(tok(response,add_special_tokens=False).input_ids)>MAX_RESPONSE_TOKENS:
                ex["response_over_max_tokens"]+=1;continue
            seen.add(h);kept.append(row)
        out[split]=kept;excluded[split]=dict(ex)
    dev,train=out["test"],out["train"]
    write(out_root/"dev.jsonl",dev)
    write(out_root/"train.jsonl",train)
    validate_holdout(train,protected+dev)
    report={"source":SOURCE,"revision":REVISION,"license":"apache-2.0",
      "parquet_sha256":{s:hashlib.sha256((upstream_root/f"upstream/data/smol-constraints/{s}-00000-of-00001.parquet").read_bytes()).hexdigest() for s in raw},
      "raw":{s:len(v) for s,v in raw.items()},"train":len(train),"dev":len(dev),"excluded":excluded,
      "max_prompt_tokens":MAX_PROMPT_TOKENS,"max_response_tokens":MAX_RESPONSE_TOKENS,
      "leakage_check":"same checks as the OPD smol_constraints pool: normalized exact vs IFEval/dev/"
                      "Tulu3 dev/Tulu3 train, and 8-gram content vs IFEval excluding n-grams in >=5 prompts; "
                      "semantic paraphrase not ruled out",
      "answer_leak_check":"dropped prompts sharing any normalized 12-gram with their own reference answer",
      "note":"kept the assistant response as the SFT training target; the sibling OPD pool at "
             "data/instruction/smol_constraints/ drops it on purpose so a frozen teacher never sees it"}
    (out_root/"preparation_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False))
if __name__=="__main__":main()
