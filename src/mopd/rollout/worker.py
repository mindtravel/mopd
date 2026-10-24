"""Persistent one-GPU rollout worker; newline JSON protocol over stdio."""

import argparse, json, sys, traceback
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from mopd.rollout.hf_batch import generate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    ).to(a.device)
    model = get_peft_model(
        base,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.requires_grad_(False)
    model.eval()
    print(json.dumps({"ready": True, "device": a.device}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("stop"):
            break
        try:
            state = torch.load(request["state"], map_location="cpu", weights_only=True)
            target = {n: p for n, p in model.named_parameters() if "lora_" in n}
            if set(state) != set(target):
                raise ValueError("LoRA structure mismatch")
            with torch.no_grad():
                for n, p in target.items():
                    p.copy_(state[n].to(a.device))
                for n, p in target.items():
                    if not torch.equal(p, state[n].to(a.device)):
                        raise ValueError("LoRA sync mismatch")
            torch.manual_seed(request["seed"])
            torch.cuda.synchronize(a.device)
            torch.cuda.reset_peak_memory_stats(a.device)
            output, seconds = generate(
                model,
                tok,
                request["rows"],
                a.device,
                request["max_new_tokens"],
                request["batch_size"],
                prompt_format=request["prompt_format"],
                sample=request["sample"],
                min_new_tokens=request["min_new_tokens"],
            )
            result = {
                "output": output,
                "version": request["version"],
                "seconds": seconds,
                "device": a.device,
                "peak_gib": torch.cuda.max_memory_allocated(a.device) / 2**30,
            }
            print(json.dumps(result), flush=True)
        except BaseException:
            print(json.dumps({"error": traceback.format_exc()}), flush=True)
            raise


if __name__ == "__main__":
    main()
