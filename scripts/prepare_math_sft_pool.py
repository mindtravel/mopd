"""Build a math SFT pool with reference solutions, reusing the exact same 1024
NuminaMath problems scripts/prepare_numina.py already selected for MOPD training.

data/numina/train.jsonl only keeps {id, source, problem}: prepare_numina.py reads the
raw parquet with columns=["source","problem"] on purpose, because an OPD teacher must
never see the reference solution. Here the solution is the SFT target, so it is read
back from the same raw parquet row the id points at ("numina-shard0:<row index>").
Reusing the identical problem set means a math-only SFT run and the MOPD runs trained
on data/numina/train.jsonl differ only in training method, not in which problems were
trained on.
"""
import json
from pathlib import Path
import pyarrow.parquet as pq
from mopd.data.pool import load_pool,text_hash,validate_holdout
from mopd.protocol import student_prompt
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]
MAX_PROMPT_TOKENS=1024
MAX_RESPONSE_TOKENS=1024
def main():
    out_root=ROOT/"data/sft/numina"
    tok=AutoTokenizer.from_pretrained(str(ROOT/"models/Qwen3-4B-Base"),local_files_only=True)
    selected=load_pool(ROOT/"data/numina/train.jsonl")  # [{id:"numina-shard0:<i>", source, problem}, ...]
    solutions=pq.read_table(ROOT/"data/numina/train-00000.parquet",columns=["solution"]).to_pylist()
    held_out=load_pool(ROOT/"data/math/eval.jsonl")  # 128 official MATH test problems, used as the shared benchmark
    excluded={}
    kept=[];dropped_prompt=0;dropped_response=0
    for row in selected:
        index=int(row["id"].split(":")[-1])
        response=solutions[index]["solution"]
        sft_row={"id":row["id"],"domain":"math","source":row["source"],"prompt":row["problem"],
                 "response":response,"split":"train","metadata":{"numina_row_index":index}}
        if len(tok(student_prompt(tok,sft_row,"plain_qa_v1")).input_ids)>MAX_PROMPT_TOKENS:
            dropped_prompt+=1;continue
        if len(tok(response,add_special_tokens=False).input_ids)>MAX_RESPONSE_TOKENS:
            dropped_response+=1;continue
        kept.append(sft_row)
    excluded={"prompt_over_max_tokens":dropped_prompt,"response_over_max_tokens":dropped_response}
    validate_holdout(kept,[{"prompt":r["problem"]} for r in held_out])
    out_root.mkdir(parents=True,exist_ok=True)
    (out_root/"train.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in kept))
    report={"source":"data/numina/train.jsonl (NuminaMath olympiads/amc_aime, seed 20261013 selection)",
      "solution_source":"data/numina/train-00000.parquet:solution, same row index as the selected id",
      "selected":len(selected),"train":len(kept),"excluded":excluded,
      "max_prompt_tokens":MAX_PROMPT_TOKENS,"max_response_tokens":MAX_RESPONSE_TOKENS,
      "shared_eval_set":"data/math/eval.jsonl (128 held-out MATH test problems); disjoint from train, checked via validate_holdout",
      "note":"identical 1024 problems scripts/prepare_numina.py selected for MOPD math training; "
             "only the training target differs (real solution vs teacher-scored student rollout)"}
    (out_root/"preparation_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False))
if __name__=="__main__":main()
