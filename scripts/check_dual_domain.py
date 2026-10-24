"""Read-only progress summary for dual-domain training."""

from mopd import paths
import json
from pathlib import Path

root = paths.ROOT
for name in [
    "logs/dual-domain-supervisor-status.json",
    "runs/20261013-dual-domain-batch32-mgpu-smoke/status.json",
    "runs/20261013-dual-domain-batch32-mgpu-opd/status.json",
]:
    p = root / name
    if p.exists():
        print(name + "\n" + p.read_text())
p = root / "runs/20261013-dual-domain-batch32-mgpu-opd/metrics.jsonl"
if p.exists():
    rows = p.read_text().splitlines()
    if rows:
        r = json.loads(rows[-1])
        print("Latest completed step:", r["step"])
        print("rollout tokens/s:", round(r["generated_tokens_per_second"], 1))
        for domain, s in r["domains"].items():
            print(domain, s)
