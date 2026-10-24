"""Linear interpolation of a fine-tuned full model with its base (WiSE-FT): w = base + alpha * (ft - base).

  python scripts/interpolate_models.py --base models/Qwen3-4B-Base --model runs/.../checkpoint-0128 \
      --alpha 0.5 --out models/interp/<name>-a0.5

alpha 1 is the fine-tuned model, 0 the base. bf16 in and out, arithmetic in fp32, on CPU; the output
directory is finalized against the base (mopd.runtime.model_files) so vLLM loads it like the base.
"""

import argparse, glob, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from mopd.runtime.model_files import finalize


def tensors(d):
    out = {}
    for f in sorted(glob.glob(str(Path(d) / "*.safetensors"))):
        with safe_open(f, "pt") as h:
            for k in h.keys():
                out[k] = (f, k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    base, ft = tensors(a.base), tensors(a.model)
    ft.pop("lm_head.weight", None) if "lm_head.weight" not in base else None
    if set(base) != set(ft):
        raise ValueError(f"tensor names differ: {sorted(set(base) ^ set(ft))[:5]}")
    merged = {}
    for k in base:
        with safe_open(base[k][0], "pt") as h:
            b = h.get_tensor(k).float()
        with safe_open(ft[k][0], "pt") as h:
            f = h.get_tensor(k).float()
        merged[k] = (b + a.alpha * (f - b)).to(torch.bfloat16)
    save_file(merged, str(out / "model.safetensors"), metadata={"format": "pt"})
    finalize(out, a.base)
    (out / "interpolation.json").write_text(json.dumps({"base": a.base, "model": a.model, "alpha": a.alpha}, indent=1))
    print(f"{out}: alpha={a.alpha}, {len(merged)} tensors")


if __name__ == "__main__":
    main()
