"""Import pinned Tulu Persona IF; reuse the standard prompt-pool converter."""
import collections,hashlib,json,random,subprocess,sys
from pathlib import Path
import pyarrow.parquet as pq
from mopd.data.pool import text_hash,validate_holdout,build_plan,load_pool
ROOT=Path(__file__).resolve().parents[1]
def write(path,rows):
    with path.open("x") as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+"\n")
def main():
    root=ROOT/"data/instruction/tulu3_if"
    raw=pq.read_table(root/"train.parquet").to_pylist()
    protected=load_pool(ROOT/"data/instruction/ifeval.jsonl")+load_pool(ROOT/"data/instruction/dev.jsonl")
    forbidden={text_hash(r["prompt"]) for r in protected};seen=set();kept=[];excluded=collections.Counter()
    for r in raw:
        h=text_hash(r["prompt"])
        if h in forbidden:excluded["holdout_exact"]+=1;continue
        if h in seen:excluded["duplicate_prompt"]+=1;continue
        seen.add(h);kept.append(r)
    random.Random(20261014).shuffle(kept)
    dev,train=kept[:512],kept[512:]
    write(root/"dev.jsonl",[dict(r,domain="instruction",split="dev",source="allenai/tulu-3-sft-personas-instruction-following") for r in dev])
    write(root/"train_source.jsonl",train)
    write(root/"protected.jsonl",protected+dev)
    subprocess.run([sys.executable,str(ROOT/"scripts/prepare_instruction_pool.py"),"--input",str(root/"train_source.jsonl"),"--holdout",str(root/"protected.jsonl"),"--source","allenai/tulu-3-sft-personas-instruction-following","--output",str(root/"train.jsonl")],check=True)
    converted=load_pool(root/"train.jsonl")
    build_plan(converted,seed=20261014,steps=64,batch_size=16)
    report={"raw":len(raw),"train":len(train),"dev":len(dev),"excluded":dict(excluded),
      "constraint_counts":dict(collections.Counter(c for r in train for c in r["constraints"])),
      "constraints_per_prompt":dict(collections.Counter(len(r["constraints"]) for r in train)),
      "parquet_sha256":hashlib.sha256((root/"train.parquet").read_bytes()).hexdigest(),
      "revision":json.loads((root/"upstream_metadata.json").read_text())["sha"],
      "leakage_check":"normalized exact only; semantic paraphrase contamination not ruled out"}
    (root/"preparation_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False))
if __name__=="__main__":main()
