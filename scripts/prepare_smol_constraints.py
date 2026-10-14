"""Import pinned HuggingFaceTB/smoltalk smol-constraints and merge it with Tulu3 IF into one instruction pool.

Only the user prompt is trained on. Upstream train/test is kept: test becomes our dev holdout.
Some upstream prompts carry (part of) the reference answer after the constraints; those are
dropped because the teacher would see the answer while scoring the student.
"""
import collections,hashlib,json,re
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from mopd.data.pool import text_hash,validate_holdout,build_plan,load_pool
from mopd.protocol import student_prompt
ROOT=Path(__file__).resolve().parents[1]
SOURCE="HuggingFaceTB/smoltalk:smol-constraints"
REVISION="5feaf2fd3ffca7c237fc38d1861bc30365d48ffa"
MAX_PROMPT_TOKENS=1024  # vllm_worker max_model_len 9216 minus max_new_tokens 8192
SEED=20261014
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
    root=ROOT/"data/instruction/smol_constraints"
    tok=AutoTokenizer.from_pretrained(str(ROOT/"models/Qwen3-4B-Base"),local_files_only=True)
    protected=load_pool(ROOT/"data/instruction/tulu3_if/protected.jsonl")
    tulu=load_pool(ROOT/"data/instruction/tulu3_if/train.jsonl")
    forbidden={text_hash(r["prompt"]) for r in protected}
    tulu_hashes={text_hash(r["prompt"]) for r in tulu}
    raw={s:load_split(root,s) for s in ("train","test")}
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
            prompt=m[0]["content"];h=text_hash(prompt)
            if h in forbidden:ex["holdout_exact"]+=1;continue
            if h in tulu_hashes:ex["tulu3_train_exact"]+=1;continue
            if h in seen:ex["duplicate_prompt"]+=1;continue
            if answer_in_prompt(prompt,m[1]["content"]):ex["reference_answer_in_prompt"]+=1;continue
            if grams(prompt)&ifeval_grams:ex["ifeval_content_8gram"]+=1;continue
            row={"id":f"smol_constraints-{split}-{i}","domain":"instruction","source":SOURCE,"prompt":prompt,
                 "split":"dev" if split=="test" else "train","metadata":{"upstream_split":split,"upstream_index":i}}
            if len(tok(student_prompt(tok,row,"plain_qa_v1")).input_ids)>MAX_PROMPT_TOKENS:
                ex["prompt_over_max_tokens"]+=1;continue
            seen.add(h);kept.append(row)
        out[split]=kept;excluded[split]=dict(ex)
    dev,train=out["test"],out["train"]
    write(root/"dev.jsonl",dev)
    write(root/"train.jsonl",train)
    validate_holdout(train,protected+dev)
    mixed=tulu+train
    build_plan(mixed,seed=SEED,steps=64,batch_size=16)
    validate_holdout(mixed,protected+dev)
    pool=ROOT/"data/instruction/mixed_tulu3_smol_constraints/train.jsonl"
    write(pool,mixed)
    report={"source":SOURCE,"revision":REVISION,"license":"apache-2.0",
      "parquet_sha256":{s:hashlib.sha256((root/f"upstream/data/smol-constraints/{s}-00000-of-00001.parquet").read_bytes()).hexdigest() for s in raw},
      "raw":{s:len(v) for s,v in raw.items()},"train":len(train),"dev":len(dev),"excluded":excluded,
      "mixed_pool":{"path":str(pool),"count":len(mixed),"by_source":dict(collections.Counter(r["source"] for r in mixed))},
      "max_prompt_tokens":MAX_PROMPT_TOKENS,
      "leakage_check":"upstream states decontamination against IFEval; additionally normalized exact vs IFEval/dev/"
                      "Tulu3 dev/Tulu3 train, and 8-gram content vs IFEval excluding n-grams in >=5 prompts; "
                      "semantic paraphrase not ruled out",
      "answer_leak_check":"dropped prompts sharing any normalized 12-gram with their reference answer"}
    (root/"preparation_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False))
if __name__=="__main__":main()
