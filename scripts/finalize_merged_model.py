"""Make a merged (or exported) model directory load exactly like its base under every env.

Writers under transformers 5 (mergekit, peft merge) re-serialize config.json in a format
the vLLM env (transformers 4.55) misreads: rope_theta moves into rope_parameters, so the
old reader falls back to theta 10000 and the model degrades badly without any error
(observed: merged HumanEval 32% vs base 85%). LoRA/merges never change architecture, so
the base's config, generation config and tokenizer files are copied byte-for-byte and
every tensor shape is asserted equal to the base's; any mismatch (e.g. a vocab resize)
fails here instead of silently in evaluation.
"""
import argparse,json,shutil,sys
from pathlib import Path
from safetensors import safe_open
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from mopd.runtime.model_files import BASE_FILES as FILES  # one list for every model directory we write
def shapes(d):
    d=Path(d);index=d/"model.safetensors.index.json"
    files=sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    out={}
    for f in files:
        with safe_open(str(d/f),"pt") as h:
            for k in h.keys():out[k]=tuple(h.get_slice(k).get_shape())
    return out
def main():
    p=argparse.ArgumentParser();p.add_argument("--base",required=True);p.add_argument("model");a=p.parse_args()
    base,model=Path(a.base),Path(a.model)
    b,m=shapes(base),shapes(model)
    if set(b)!=set(m):raise ValueError(f"tensor names differ: {sorted(set(b)^set(m))[:5]}")
    bad={k:(b[k],m[k]) for k in b if b[k]!=m[k]}
    if bad:raise ValueError(f"tensor shapes differ from base: {list(bad.items())[:5]}")
    moved=model/"written-by-merge";moved.mkdir(exist_ok=True)
    for f in FILES+("chat_template.jinja",):
        if (model/f).exists():shutil.move(str(model/f),str(moved/f))
    for f in FILES:
        if (base/f).exists():shutil.copy(base/f,model/f)
    print(f"{model}: {len(m)} tensors match base shapes; base config/tokenizer copied")
if __name__=="__main__":main()
