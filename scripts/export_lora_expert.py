"""Fold one SFT LoRA checkpoint into its base and save a standalone bf16 model.

mergekit merges full weight sets, so each domain expert (base + LoRA) is materialized
once as a regular HF model directory. Runs on CPU so no GPU is held. Records the
adapter sha256 and source run so a merged model can be traced back to its experts.
"""
import argparse,hashlib,json,shutil
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
def main():
    p=argparse.ArgumentParser();p.add_argument("--base",required=True);p.add_argument("--adapter",required=True)
    p.add_argument("--out",required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    model=AutoModelForCausalLM.from_pretrained(a.base,dtype=torch.bfloat16,local_files_only=True)
    model=PeftModel.from_pretrained(model,a.adapter).merge_and_unload()
    model.save_pretrained(out,safe_serialization=True)
    # Copy the base's tokenizer files byte-for-byte: re-saving with transformers 5 writes a
    # tokenizer_config that the vLLM env (transformers 4.55) cannot load; the vocabulary is identical.
    for name in ("tokenizer.json","tokenizer_config.json","vocab.json","merges.txt","generation_config.json"):
        if (Path(a.base)/name).is_file():shutil.copy(Path(a.base)/name,out/name)
    adapter=Path(a.adapter)
    (out/"expert-source.json").write_text(json.dumps({"base":str(Path(a.base).resolve()),"adapter":str(adapter.resolve()),
        "adapter_sha256":hashlib.sha256((adapter/"adapter_model.safetensors").read_bytes()).hexdigest(),
        "run_config":json.loads((adapter.parent/"config.json").read_text())},indent=2))
    print("saved",out)
if __name__=="__main__":main()
