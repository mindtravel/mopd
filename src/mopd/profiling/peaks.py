"""Measured hardware peaks per GPU model (bf16 matmul TFLOPS, device-memory copy GB/s), cached in
cache/profile/ so every report divides by the same numbers. Reading needs no torch.

  python -m mopd.profiling.peaks --measure [--device 6]     # once, on an idle GPU
"""

import json
import subprocess
import time

from mopd import paths

CACHE = paths.PROFILE_CACHE


def gpu_name(index=None):
    """Name as torch and nvidia-smi both report it ("NVIDIA GeForce RTX 5090"), without importing torch."""
    cmd = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"] + (["-i", str(index)] if index is not None else [])
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.splitlines()[0].strip()
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None


def cached_peaks(index=None):
    """{"bf16_tflops", "hbm_gbps"} measured for this GPU model, or {} if never measured."""
    name = gpu_name(index)
    f = CACHE / f"peak-{(name or '').replace(' ', '_')}.json"
    return json.loads(f.read_text()) if name and f.is_file() else {}


def measure_peaks(device=0, n=8192, iters=50, copy_gib=2):
    """Dense bf16 matmul TFLOPS and device-memory copy GB/s (read + write) of one GPU; cached."""
    import torch

    torch.cuda.set_device(device)
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)

    def timed(fn, reps):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return time.perf_counter() - t

    tflops = 2 * n**3 * iters / timed(lambda: a @ b, iters) / 1e12
    del a, b
    src = torch.empty(copy_gib << 30, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    gbps = 2 * src.numel() * 20 / timed(lambda: dst.copy_(src), 20) / 1e9
    name = torch.cuda.get_device_name()
    CACHE.mkdir(parents=True, exist_ok=True)
    peaks = {"gpu": name, "bf16_tflops": round(tflops, 1), "hbm_gbps": round(gbps), "measured": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (CACHE / f"peak-{name.replace(' ', '_')}.json").write_text(json.dumps(peaks, indent=1))
    return peaks


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--device", type=int, default=0)
    a = ap.parse_args()
    if a.measure:
        print(json.dumps(measure_peaks(a.device)))
