"""Freeze small pilot evaluation sets, independent of all training prompts."""
import json,random
from pathlib import Path
from mopd.data.pool import validate_holdout
from train_multidomain_opd import plan
root=Path("/ssd/lzx/mopd");c=json.loads((root/"configs/dual_domain.json").read_text())
train=[r for batch in plan(c) for r in batch]
math=[dict(r,domain="math",prompt=r["problem"]) for r in map(json.loads,(root/"data/math/eval.jsonl").read_text().splitlines())]
instruction=list(map(json.loads,(root/"data/instruction/ifeval.jsonl").read_text().splitlines()))
random.Random(20261013).shuffle(math);random.Random(20261013).shuffle(instruction)
rows=math[:32]+instruction[:32];validate_holdout(train,rows)
out=root/"data/dual_eval";out.mkdir(exist_ok=True)
for name,data in [("pilot",rows),("math",math[:32]),("instruction",instruction[:32])]:
    with (out/(name+".jsonl")).open("x") as f:
        for r in data:f.write(json.dumps(r,ensure_ascii=False)+"\n")
print("frozen: 32 math + 32 IFEval; evaluation only")
