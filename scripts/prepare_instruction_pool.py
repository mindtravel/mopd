"""Import public/local JSONL prompts, preserving raw metadata; no benchmark prompts as training."""
import argparse, json, hashlib
from pathlib import Path
from mopd.data.pool import build_plan, validate_holdout

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--holdout", required=True, help="JSONL with prompt field; reserved dev/test")
    p.add_argument("--source", required=True)
    p.add_argument("--prompt-field", default="prompt")
    p.add_argument("--output", required=True)
    a=p.parse_args()
    raw=Path(a.input).read_bytes()
    rows=[]
    for i,line in enumerate(raw.decode().splitlines()):
        if not line.strip(): continue
        r=json.loads(line)
        rows.append({"id":str(r.get("id",i)), "domain":"instruction", "split":"train",
          "source":a.source, "prompt":r[a.prompt_field], "metadata":r})
    build_plan(rows,seed=0,steps=1,batch_size=1)
    holdout=[json.loads(l) for l in Path(a.holdout).read_text().splitlines() if l.strip()]
    validate_holdout(rows,holdout)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x") as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+"\n")
    out.with_suffix(".manifest.json").write_text(json.dumps({
       "source":a.source,"input_sha256":hashlib.sha256(raw).hexdigest(),
       "count":len(rows),"holdout":a.holdout,
       "overlap_check":"normalized exact only; not semantic dedup"},indent=2))
if __name__=="__main__":main()
