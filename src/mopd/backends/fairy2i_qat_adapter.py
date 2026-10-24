"""MOPD-owned compatibility adapter for Fairy2i QAT autograd."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path


def install_scale_backward_fallback(kernel_module) -> None:
    """Replace the scale backward op when the installed Triton lacks TMA descriptors."""
    import torch

    upstream_quantize = kernel_module.fairytoi_quant_block

    class QuantizeWeight(torch.autograd.Function):
        @staticmethod
        def forward(ctx, weight, scale):
            ctx.save_for_backward(weight)
            ctx.scale_shape = scale.shape
            ctx.scale_dtype = scale.dtype
            return upstream_quantize(weight, scale)

        @staticmethod
        def backward(ctx, grad_quantized_weight):
            (weight,) = ctx.saved_tensors
            m, n = weight.shape[0] // 2, weight.shape[1] // 2
            blocks_m, blocks_n = m // 64, n // 64

            def tiles(tensor):
                return tensor.reshape(blocks_m, 64, blocks_n, 64).permute(0, 2, 1, 3).float()

            w11, w12 = tiles(weight[:m, :n]), tiles(weight[:m, n:])
            w21, w22 = tiles(weight[m:, :n]), tiles(weight[m:, n:])
            g11, g12 = tiles(grad_quantized_weight[:m, :n]), tiles(grad_quantized_weight[:m, n:])
            g21, g22 = tiles(grad_quantized_weight[m:, :n]), tiles(grad_quantized_weight[m:, n:])

            u_re = 0.5 * (w11 + w22)
            u_im = 0.5 * (w21 - w12)
            w_re = 0.5 * (w11 - w22)
            w_im = 0.5 * (w12 + w21)

            def signed(value):
                return torch.where(value >= 0, 1.0, -1.0)

            u_re_dominates = u_re.abs() > u_im.abs()
            w_re_dominates = w_re.abs() > w_im.abs()
            grad_u_re = ((g11 + g22) * torch.where(u_re_dominates, signed(u_re), 0.0)).sum((-1, -2))
            grad_u_im = ((-g12 + g21) * torch.where(~u_re_dominates, signed(u_im), 0.0)).sum((-1, -2))
            grad_w_re = ((g11 - g22) * torch.where(w_re_dominates, signed(w_re), 0.0)).sum((-1, -2))
            grad_w_im = ((g12 + g21) * torch.where(~w_re_dominates, signed(w_im), 0.0)).sum((-1, -2))
            grad_scale = torch.stack((grad_u_re, grad_u_im, grad_w_re, grad_w_im)).to(ctx.scale_dtype)
            if tuple(grad_scale.shape) != tuple(ctx.scale_shape):
                raise RuntimeError(f"scale gradient shape mismatch: {grad_scale.shape} != {ctx.scale_shape}")

            return grad_quantized_weight, grad_scale

    kernel_module.QATLinearComplex_quant = lambda weight, scale: QuantizeWeight.apply(weight, scale)


def _scale_parameters(model):
    return {name: parameter for name, parameter in model.named_parameters() if name.endswith(".quant_scale")}


def save_fairy2i_qat_update(model, output_dir, *, base_model, step=None):
    from safetensors.torch import save_file

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.endswith(".quant_scale")
    ]
    if unexpected:
        raise ValueError(f"Only quant_scale parameters may be trainable: {unexpected[:3]}")
    state = {
        name: parameter.detach().to(device="cpu").contiguous().clone()
        for name, parameter in _scale_parameters(model).items()
    }
    if not state:
        raise ValueError("Fairy2i QAT model has no quant_scale parameters")

    target = output / "qat_scales.safetensors"
    temporary = target.with_suffix(".safetensors.tmp")
    save_file(state, str(temporary))
    os.replace(temporary, target)

    manifest = {
        "schema": "ifairy.fairy2i_qat_update.v1",
        "base_model": str(base_model),
        "update_type": "quant_scale_only",
        "step": step,
        "tensor_count": len(state),
        "parameter_count": sum(tensor.numel() for tensor in state.values()),
        "tensors": {name: list(tensor.shape) for name, tensor in state.items()},
    }
    manifest_path = output / "qat-update.json"
    partial = manifest_path.with_suffix(".json.tmp")
    partial.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(partial, manifest_path)
    return target


def load_fairy2i_qat_update(model, update_path, *, expected_base_model=None):
    from safetensors.torch import load_file
    import torch

    path = Path(update_path)
    if path.is_dir():
        manifest_path = path / "qat-update.json"
        path = path / "qat_scales.safetensors"
    else:
        manifest_path = path.with_name("qat-update.json")
    if not path.is_file():
        raise FileNotFoundError(f"Missing Fairy2i QAT scale update: {path}")
    if expected_base_model and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("base_model") != str(expected_base_model):
            raise ValueError("Fairy2i QAT update was saved for a different base model")

    expected = _scale_parameters(model)
    state = load_file(str(path), device="cpu")
    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        extra = sorted(set(state) - set(expected))
        raise ValueError(f"Fairy2i quant_scale keys mismatch; missing={missing[:3]}, extra={extra[:3]}")
    with torch.no_grad():
        for name, parameter in expected.items():
            if tuple(state[name].shape) != tuple(parameter.shape):
                raise ValueError(f"Fairy2i quant_scale shape mismatch for {name}")
            parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))
    return model


def load_fairy2i_qat_student(
    model_path,
    device,
    *,
    trainable_scales=True,
    update_path=None,
):
    """Load the custom 1-bit forward model without changing its read-only source."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = Path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(source),
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).to(device)
    qat_layer = next(
        (module for module in model.modules() if type(module).__name__ == "QATLinearComplex"),
        None,
    )
    if qat_layer is None:
        raise ValueError(f"Model has no Fairy2i QAT linear layers: {source}")
    kernel_module = importlib.import_module(type(qat_layer).__module__)
    install_scale_backward_fallback(kernel_module)

    if update_path:
        load_fairy2i_qat_update(model, update_path, expected_base_model=source)

    model.requires_grad_(False)
    scales = _scale_parameters(model)
    if not scales:
        raise ValueError(f"Model has no Fairy2i quant_scale parameters: {source}")
    for parameter in scales.values():
        parameter.requires_grad_(trainable_scales)
    return tokenizer, model
