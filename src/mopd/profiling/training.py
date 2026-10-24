"""Training-side profiling (torch): per-step timers with MFU/HFU, and a torch.profiler kernel
breakdown of one chosen step (config "profile_step")."""

import json
import time
from contextlib import contextmanager
from pathlib import Path

from mopd.profiling.peaks import cached_peaks


class StepProfile:
    """Per-step timers and token accounting for a trainer. Usage:
    prof = StepProfile(shape, gpus, recompute=gc_enabled)
    with prof.phase("forward_backward"): ...
    prof.add_batch(lengths, padded_width)
    report = prof.finish()      # dict for the step's metrics line
    """

    def __init__(self, shape, gpus, *, recompute, peak_tflops=None, device=None):
        self.shape, self.gpus, self.recompute, self.device = shape, gpus, recompute, device
        self.peak = peak_tflops if peak_tflops is not None else cached_peaks().get("bf16_tflops")
        self.phases, self.lengths, self.padded, self.start = {}, [], 0, time.perf_counter()

    @contextmanager
    def phase(self, name):
        import torch

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
        tflops = self.shape.train_flops(tokens, sum_sq) / seconds / self.gpus / 1e12
        report = {
            "seconds": round(seconds, 3),
            "phases": {k: round(v, 3) for k, v in self.phases.items()},
            "tokens": int(tokens),
            "padding_fraction": round(1 - tokens / padded, 4) if padded else 0.0,
            "model_tflops_per_gpu": round(tflops, 2),
            "peak_tflops": self.peak,
        }
        if self.peak:
            report["mfu"] = round(tflops / self.peak, 4)
            hw = self.shape.train_flops(tokens, sum_sq, recompute=self.recompute)
            report["hfu"] = round(hw / seconds / self.gpus / 1e12 / self.peak, 4)
        return report


KERNEL_CLASSES = (
    ("communication", ("nccl", "ncclKernel", "AllGather", "ReduceScatter", "AllReduce")),
    ("matmul", ("gemm", "cutlass", "matmul", "Gemm", "cublas")),
    ("attention", ("attention", "flash", "fmha", "sdp")),
)


def trace_summary(prof, wall_seconds):
    """CUDA kernel time of a torch.profiler trace by class (communication / matmul / attention / other)
    and the share of wall time with no kernel running. Streams overlap, so classes can exceed 100%."""
    totals = {name: 0.0 for name, _ in KERNEL_CLASSES} | {"other": 0.0}
    for e in prof.key_averages():
        if getattr(e, "device_time_total", 0) <= 0 or e.device_type.name != "CUDA":
            continue
        cls = next((name for name, keys in KERNEL_CLASSES if any(k.lower() in e.key.lower() for k in keys)), "other")
        totals[cls] += e.device_time_total / 1e6
    return {
        "wall_seconds": round(wall_seconds, 3),
        "kernel_seconds": {k: round(v, 3) for k, v in totals.items()},
        "kernel_share_of_wall": {k: round(v / wall_seconds, 3) for k, v in totals.items()},
        "idle_share_of_wall": round(max(0.0, 1 - sum(totals.values()) / wall_seconds), 3),
    }


@contextmanager
def torch_profile(enabled, out_dir, step):
    """torch.profiler around one step when enabled (config "profile_step"): writes
    profile-step-<N>.json (trace_summary) and .trace.json.gz under out_dir; yields that summary."""
    result = {}
    if not enabled:
        yield result
        return
    import torch
    from torch.profiler import ProfilerActivity, profile

    start = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        yield result
        torch.cuda.synchronize()
    result.update(trace_summary(prof, time.perf_counter() - start))
    out = Path(out_dir)
    (out / f"profile-step-{step:04d}.json").write_text(json.dumps(result, indent=1))
    prof.export_chrome_trace(str(out / f"profile-step-{step:04d}.trace.json.gz"))
