"""Load a Fairy2i checkpoint, generate, and verify trainable scale gradients."""
import argparse
import json
import os
from pathlib import Path

import torch

from mopd.backends.fairy2i_qat_adapter import load_fairy2i_qat_student
from mopd.data.templates import apply_chat_template_ids
from mopd.training.batching import generation_eos_ids


ROOT = Path("/ssd/lzx/mopd")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    report_path = Path(args.report).resolve()
    if not report_path.is_relative_to(ROOT):
        raise ValueError("Report must stay inside /ssd/lzx/mopd")

    tokenizer, model = load_fairy2i_qat_student(args.model, args.device)
    eos_ids = generation_eos_ids(model, tokenizer)
    messages = [{"role": "user", "content": "Write one sentence about unit tests."}]
    input_ids = apply_chat_template_ids(
        tokenizer, messages, add_generation_prompt=True, return_tensors="pt"
    ).to(args.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=list(eos_ids),
        )

    model.train()
    model.config.use_cache = False
    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    loss = logits[:, -1, :].float().square().mean()
    loss.backward()
    scales = [
        parameter
        for name, parameter in model.named_parameters()
        if name.endswith(".quant_scale")
    ]
    scale_grad_norm = sum(
        parameter.grad.float().square().sum()
        for parameter in scales
        if parameter.grad is not None
    ).sqrt()
    result = {
        "model": str(Path(args.model).resolve()),
        "model_class": type(model).__module__ + "." + type(model).__name__,
        "transformers_version": __import__("transformers").__version__,
        "generated_tokens": int(generated.shape[-1] - input_ids.shape[-1]),
        "answer": tokenizer.decode(generated[0, input_ids.shape[-1] :], skip_special_tokens=False),
        "trainable_scale_tensors": len(scales),
        "trainable_scale_parameters": sum(parameter.numel() for parameter in scales),
        "loss_finite": bool(torch.isfinite(loss)),
        "scale_grad_finite": all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in scales
        ),
        "scale_grad_norm": float(scale_grad_norm),
    }
    if not result["loss_finite"] or not result["scale_grad_finite"] or result["scale_grad_norm"] <= 0:
        raise RuntimeError("Fairy2i QAT forward/backward smoke failed: " + json.dumps(result))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    partial = report_path.with_suffix(report_path.suffix + ".tmp")
    partial.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    os.replace(partial, report_path)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
