"""Larger Numina math SFT pool (problem + reference solution) for budget-matched runs.

A budget-matched math-only SFT run needs 64 steps x 32 = 2048 distinct problems; the
first SFT pool (data/sft/numina, 884 rows) was just MOPD's 1024 selected problems
minus long solutions. This continues scripts/prepare_numina.py's exact selection loop
(same eligibility, same seed-20261013 shuffle, same exact + near-duplicate screen
against all 5000 MATH test problems) past 1024 up to TARGET, so its first 1024
selections are by construction MOPD's data/numina/train.jsonl, asserted below.
prepare_numina.py runs at import time, so the loop is restated rather than imported.

Long solutions are kept up to MAX_RESPONSE_TOKENS (SFT max_seq_len 4096 minus the
prompt cap), instead of the first pool's 1024 cap that dropped 14% of the problems.
"""
import collections,difflib,hashlib,json,random,re
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from mopd.data.pool import load_pool
from mopd.protocol import student_prompt
ROOT=Path(__file__).resolve().parents[1]
TARGET=4096
MAX_PROMPT_TOKENS=1024
MAX_RESPONSE_TOKENS=3072
norm=lambda x:re.sub(r"[^a-z0-9]","",x.lower())
def grams(s):return {s[i:i+5] for i in range(max(0,len(s)-4))}
def select():
    test=[]
    for p in (ROOT/"data/math/raw").glob("*/test-*.parquet"):test.extend(pq.read_table(p).to_pylist())
    keys={norm(r["problem"]) for r in test}
    testnorm=[norm(r["problem"]) for r in test];tg=[grams(s) for s in testnorm]
    inv=collections.defaultdict(list)
    for i,g in enumerate(tg):
        for v in g:inv[v].append(i)
    rows=pq.read_table(ROOT/"data/numina/train-00000.parquet",columns=["source","problem","solution"]).to_pylist()
    eligible=[(i,r) for i,r in enumerate(rows) if r["source"] in ["olympiads","amc_aime"] and "[asy]" not in r["problem"] and 80<=len(r["problem"])<=3000]
    random.Random(20261013).shuffle(eligible)
    selected=[];seen=set();excluded=collections.Counter()
    for i,r in eligible:
        s=norm(r["problem"])
        if s in seen or s in keys:excluded["exact"]+=1;continue
        g=grams(s);counts=collections.Counter(j for v in g for j in inv.get(v,[]))
        near=False
        for j,count in counts.most_common(5):
            jac=count/max(1,len(g|tg[j]))
            ratio=difflib.SequenceMatcher(None,s,testnorm[j],autojunk=False).ratio()
            if jac>=.65 or ratio>=.8:near=True;break
        if near:excluded["near"]+=1;continue
        seen.add(s);selected.append((i,r))
        if len(selected)==TARGET:break
    return selected,excluded,len(test)
def main():
    out_root=ROOT/"data/sft/numina_v2"
    tok=AutoTokenizer.from_pretrained(str(ROOT/"models/Qwen3-4B-Base"),local_files_only=True)
    selected,excluded,checked=select()
    mopd_ids=[r["id"] for r in load_pool(ROOT/"data/numina/train.jsonl")]
    if [f"numina-shard0:{i}" for i,_ in selected[:len(mopd_ids)]]!=mopd_ids:
        raise ValueError("selection prefix differs from data/numina/train.jsonl; the restated loop drifted from prepare_numina.py")
    kept=[];length=collections.Counter()
    for rank,(i,r) in enumerate(selected):
        row={"id":f"numina-shard0:{i}","domain":"math","source":r["source"],"prompt":r["problem"],
             "response":r["solution"],"split":"train","metadata":{"numina_row_index":i,"selection_rank":rank,
             "in_mopd_pool":rank<len(mopd_ids)}}
        if len(tok(student_prompt(tok,row,"plain_qa_v1")).input_ids)>MAX_PROMPT_TOKENS:length["prompt_over_max_tokens"]+=1;continue
        if len(tok(r["solution"],add_special_tokens=False).input_ids)>MAX_RESPONSE_TOKENS:length["response_over_max_tokens"]+=1;continue
        kept.append(row)
    out_root.mkdir(parents=True,exist_ok=True)
    path=out_root/"train.jsonl"
    path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in kept))
    report={"source":"AI-MO/NuminaMath-CoT shard0 (data/numina/train-00000.parquet), olympiads+amc_aime",
      "selection":"restated scripts/prepare_numina.py loop, seed 20261013, extended to "+str(TARGET),
      "selected":len(selected),"train":len(kept),"from_mopd_pool":sum(r["metadata"]["in_mopd_pool"] for r in kept),
      "selection_excluded":dict(excluded),"length_excluded":dict(length),"benchmark_checked":checked,
      "max_prompt_tokens":MAX_PROMPT_TOKENS,"max_response_tokens":MAX_RESPONSE_TOKENS,
      "prefix_check":"first 1024 selections equal data/numina/train.jsonl ids (MOPD math pool)",
      "limitation":"heuristic near-duplicate screen; semantic paraphrase and pretraining contamination not ruled out",
      "train_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    (out_root/"preparation_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False))
if __name__=="__main__":main()
