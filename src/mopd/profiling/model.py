"""FLOP and byte arithmetic for training and inference utilization (no torch: imported by the bench
service and the trainers alike).

Training FLOPs   = 6 N T + 6 L d_attn sum_i s_i^2      (8 N T with recomputation: HFU)
Inference FLOPs  = 2 N (generated + prefilled tokens) + 4 L d_attn (attention context read)
Decode bytes     = engine steps x weight bytes  (+ KV read when the context is known)
    N parameters (a tied lm_head once), T real tokens, L layers, d_attn query heads x head dim,
    s_i sequence lengths. MFU = FLOPs / (seconds x GPUs x peak bf16 matmul); MBU = bytes / (seconds x
    GPUs x peak copy bandwidth). Decode is bandwidth-bound: MBU and the mean tokens per engine step say
    whether the GPU is fed; MFU alone looks low there by construction.
"""

import json
from pathlib import Path

from mopd.profiling.peaks import cached_peaks


class ModelShape:
    """The numbers the FLOP/byte counts need: parameters, layers, attention and KV widths."""

    def __init__(self, config, params):
        get = (lambda k, d=None: config.get(k, d)) if isinstance(config, dict) else (lambda k, d=None: getattr(config, k, d))
        heads = get("num_attention_heads")
        head_dim = get("head_dim") or get("hidden_size") // heads
        self.params, self.layers = params, get("num_hidden_layers")
        self.attn_dim = heads * head_dim
        self.kv_bytes_per_token = 2 * self.layers * get("num_key_value_heads", heads) * head_dim * 2  # K and V, bf16

    @classmethod
    def of(cls, model):
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        base = getattr(base, "base", base)  # mopd.training.batching.TokenLogProbs
        # parameters() de-duplicates shared tensors (a tied lm_head once); numel of an FSDP shard is the full size
        return cls(base.config, sum(p.numel() for p in base.parameters()))

    @classmethod
    def of_checkpoint(cls, path):
        """From config.json and the safetensors headers only (no weights read); a LoRA adapter uses its base."""
        path = Path(path)
        if (path / "adapter_config.json").exists():
            path = Path(json.loads((path / "adapter_config.json").read_text())["base_model_name_or_path"])
        params = 0
        for f in path.glob("*.safetensors"):
            with f.open("rb") as h:
                header = json.loads(h.read(int.from_bytes(h.read(8), "little")))
            for k, v in header.items():
                if k != "__metadata__":
                    n = 1
                    for d in v["shape"]:
                        n *= d
                    params += n
        return cls(json.loads((path / "config.json").read_text()), params)

    def train_flops(self, tokens, sum_sq, recompute=False):
        """tokens: real tokens; sum_sq: sum of squared sequence lengths (causal attention, fwd+bwd)."""
        return (8 if recompute else 6) * self.params * tokens + 6 * self.layers * self.attn_dim * sum_sq


def infer_utilization(shape, *, seconds, gpus, generated, prefilled=0, steps=None, context_tokens=None, peaks=None):
    """Inference over `seconds` on `gpus`: generated / prefilled (computed, not cache-hit) tokens,
    engine steps (decode iterations) and, if known, the summed attention context read while decoding.
    Returns tokens/s, mean tokens per step, MFU and MBU (weights + KV when context_tokens is given,
    else weights only: a lower bound)."""
    peaks = peaks if peaks is not None else cached_peaks()
    seconds = max(seconds, 1e-9)
    flops = 2 * shape.params * (generated + prefilled) + 4 * shape.layers * shape.attn_dim * (context_tokens or 0)
    out = {"generated_tok_s": round(generated / seconds / gpus, 1)}
    if steps:
        out["tokens_per_step"] = round((generated + prefilled) / steps, 1)
    if peaks.get("bf16_tflops"):
        out["mfu"] = round(flops / seconds / gpus / 1e12 / peaks["bf16_tflops"], 4)
    if steps and peaks.get("hbm_gbps"):
        moved = steps * 2 * shape.params + (context_tokens or 0) * shape.kv_bytes_per_token
        out["mbu" if context_tokens else "mbu_weights"] = round(moved / seconds / gpus / 1e9 / peaks["hbm_gbps"], 4)
    return out


def opd_utilization(shape, lengths, *, seconds, gpus, rollout, recompute=True, peaks=None):
    """One MOPD step. seconds: {"step", "student", "generation"}; gpus: {"student", "rollout"};
    rollout: the engines' counters {"generated", "prefilled", "steps", "context_tokens"} summed over
    rollout GPUs. Student MFU in its phase and over the whole step (its GPUs idle while generating),
    rollout MFU/MBU and batch over the generation phase."""
    peaks = peaks if peaks is not None else cached_peaks()
    tokens, sum_sq = sum(lengths), sum(n * n for n in lengths)
    flops = shape.train_flops(tokens, sum_sq)
    student = max(seconds["student"], 1e-9)
    out = {
        "trained_tokens": tokens,
        "student_tflops_per_gpu": round(flops / student / gpus["student"] / 1e12, 2),
        "rollout_busy_fraction": round(seconds["generation"] / seconds["step"], 3) if seconds["step"] else None,
    }
    if peaks.get("bf16_tflops"):
        peak = peaks["bf16_tflops"]
        out["student_mfu_in_phase"] = round(out["student_tflops_per_gpu"] / peak, 4)
        out["student_mfu_whole_step"] = round(flops / seconds["step"] / gpus["student"] / 1e12 / peak, 4)
        out["student_hfu_in_phase"] = round(shape.train_flops(tokens, sum_sq, recompute) / student / gpus["student"] / 1e12 / peak, 4)
    r = infer_utilization(shape, seconds=seconds["generation"], gpus=max(gpus["rollout"], 1), peaks=peaks,
                          **{k: rollout.get(k) or 0 for k in ("generated", "prefilled")},
                          steps=rollout.get("steps"), context_tokens=rollout.get("context_tokens"))
    out.update({f"rollout_{k}": v for k, v in r.items()})
    return out
