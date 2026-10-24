"""Training-step profile: where a step's time goes and how much of the GPUs it uses (MFU).

One place for the arithmetic, so every trainer reports the same numbers and an optimization can be
judged against a recorded baseline:

  model FLOPs per step = 6 * N * T  +  6 * L * d_attn * sum_i s_i^2
      N: parameters (a tied embedding/lm_head counted once -- it is the lm_head matmul), T: real
      (non-padding) tokens, L: layers, d_attn: query heads * head dim, s_i: sequence lengths
      (causal attention, forward + backward = 3x forward).
  MFU = model FLOPs / (seconds * GPUs * measured peak);  HFU adds the forward recomputed by gradient
      checkpointing (8N instead of 6N on the dense part), i.e. what the hardware actually executed.

The peak is measured (a large bf16 matmul), not the datasheet number, and cached per GPU model in
cache/profile/; trainers only read the cache, since measuring on a busy GPU would understate it:

  python -m mopd.runtime.profile --measure-peak        # once, on an idle GPU
"""

import json
import time
from contextlib import contextmanager
from pathlib import Path

import torch

CACHE = Path("/ssd/lzx/mopd/cache/profile")


def _gpu_key(device=None):
    return torch.cuda.get_device_name(device).replace(" ", "_")


def measure_peak_tflops(device=None, n=8192, iters=50):
    """Achievable dense bf16 matmul throughput of one GPU, in TFLOPS; cached per GPU model."""
    if device is None:
        device = torch.cuda.current_device()  # torch.randn(device=None) would allocate on the CPU
    a = torch.randn(n, n, device=device, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=device, dtype=torch.bfloat16)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize(device)
    tflops = 2 * n**3 * iters / (time.perf_counter() - start) / 1e12
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"peak-{_gpu_key(device)}.json").write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(device),
                "bf16_tflops": round(tflops, 1),
                "matmul": n,
                "measured": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=1,
        )
    )
    return tflops


def cached_peak_tflops(device=None):
    """None when no measurement is cached for this GPU model, or without CUDA at all."""
    if not torch.cuda.is_available():
        return None
    f = CACHE / f"peak-{_gpu_key(device)}.json"
    return json.loads(f.read_text())["bf16_tflops"] if f.is_file() else None


class ModelShape:
    """The few numbers the FLOP count needs, from an HF config and the parameter count."""

    def __init__(self, config, params):
        self.params = params
        self.layers = config.num_hidden_layers
        self.attn_dim = config.num_attention_heads * getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

    @classmethod
    def of(cls, model):
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        base = getattr(base, "base", base)  # mopd.training.batching.TokenLogProbs
        # parameters() de-duplicates shared tensors (a tied lm_head appears once); numel of a
        # sharded (FSDP) parameter is its full size
        return cls(base.config, sum(p.numel() for p in base.parameters()))

    def flops(self, tokens, sum_sq, recompute=False):
        """tokens: real tokens in the step; sum_sq: sum of squared sequence lengths."""
        dense = (8 if recompute else 6) * self.params * tokens
        # QK^T and PV: 2 matmuls of 2*s^2*d_attn each per layer, causal halves them -> 2*s^2*d_attn forward
        attention = 6 * self.layers * self.attn_dim * sum_sq
        return dense + attention


class StepProfile:
    """Per-step timers and token accounting. Usage:
    prof = StepProfile(shape, gpus, recompute=gc_enabled)
    with prof.phase("forward_backward"): ...
    prof.add_batch(lengths, padded_width)
    report = prof.finish()      # dict for the step's metrics line
    """

    def __init__(self, shape, gpus, *, recompute, peak_tflops=None, device=None):
        self.shape, self.gpus, self.recompute, self.device = shape, gpus, recompute, device
        self.peak = peak_tflops if peak_tflops is not None else cached_peak_tflops(device)
        self.phases, self.lengths, self.padded, self.start = {}, [], 0, time.perf_counter()

    @contextmanager
    def phase(self, name):
        torch.cuda.synchronize(self.device)
        t = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize(self.device)
            self.phases[name] = self.phases.get(name, 0.0) + time.perf_counter() - t

    def add_batch(self, lengths, width):
        self.lengths += list(lengths)
        self.padded += width * len(lengths)

    def finish(self, total_tokens=None, padded=None, sum_sq=None):
        """Pass the whole step's totals (summed over ranks) when this rank only saw its shard."""
        tokens = sum(self.lengths) if total_tokens is None else total_tokens
        padded = self.padded if padded is None else padded
        sum_sq = sum(q * q for q in self.lengths) if sum_sq is None else sum_sq
        seconds = time.perf_counter() - self.start
        tflops_per_gpu = self.shape.flops(tokens, sum_sq) / seconds / self.gpus / 1e12
        report = {
            "seconds": round(seconds, 3),
            "phases": {k: round(v, 3) for k, v in self.phases.items()},
            "tokens": int(tokens),
            "padding_fraction": round(1 - tokens / padded, 4) if padded else 0.0,
            "model_tflops_per_gpu": round(tflops_per_gpu, 2),
            "peak_tflops": self.peak,
        }
        if self.peak:
            report["mfu"] = round(tflops_per_gpu / self.peak, 4)
            report["hfu"] = round(
                self.shape.flops(tokens, sum_sq, recompute=self.recompute) / seconds / self.gpus / 1e12 / self.peak, 4
            )
        return report


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--measure-peak", action="store_true")
    a = ap.parse_args()
    if a.measure_peak:
        print(f"{torch.cuda.get_device_name()}: {measure_peak_tflops():.1f} bf16 TFLOPS (cached in {CACHE})")


# ---------------------------------------------------------------- one-step kernel breakdown
KERNEL_CLASSES = (
    ("communication", ("nccl", "ncclKernel", "AllGather", "ReduceScatter", "AllReduce")),
    ("matmul", ("gemm", "cutlass", "matmul", "Gemm", "cublas")),
    ("attention", ("attention", "flash", "fmha", "sdp")),
)


def trace_summary(prof, wall_seconds):
    """Classify a torch.profiler trace's CUDA kernel time: communication / matmul / attention / other,
    plus the share of wall time with no kernel running (launch gaps, CPU work, host-side waits).
    Kernels overlap across streams, so classes can sum past 100% of wall time."""
    totals = {name: 0.0 for name, _ in KERNEL_CLASSES}
    totals["other"] = 0.0
    for e in prof.key_averages():
        if getattr(e, "device_time_total", 0) <= 0 or e.device_type.name != "CUDA":
            continue
        cls = next((name for name, keys in KERNEL_CLASSES if any(k.lower() in e.key.lower() for k in keys)), "other")
        totals[cls] += e.device_time_total / 1e6
    busy = sum(totals.values())
    return {
        "wall_seconds": round(wall_seconds, 3),
        "kernel_seconds": {k: round(v, 3) for k, v in totals.items()},
        "kernel_share_of_wall": {k: round(v / wall_seconds, 3) for k, v in totals.items()},
        "idle_share_of_wall": round(max(0.0, 1 - busy / wall_seconds), 3),
    }


@contextmanager
def maybe_torch_profile(enabled, out_dir, step):
    """Wrap one training step in torch.profiler when enabled; writes profile-step-<N>.json (summary)
    and profile-step-<N>.trace.json.gz (chrome trace) under out_dir. Yields a dict filled on exit."""
    result = {}
    if not enabled:
        yield result
        return
    from torch.profiler import ProfilerActivity, profile

    start = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        yield result
        torch.cuda.synchronize()
    result.update(trace_summary(prof, time.perf_counter() - start))
    out = Path(out_dir)
    (out / f"profile-step-{step:04d}.json").write_text(json.dumps(result, indent=1))
    prof.export_chrome_trace(str(out / f"profile-step-{step:04d}.trace.json.gz"))
