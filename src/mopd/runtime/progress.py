"""One atomic "how far along is this" file, written by every long-running generate/grade/train
loop so a human or another script can tell real progress from a hang without parsing stdout or
guessing from partial file sizes. Call write() as often as the loop has something new to report;
each call fully replaces the target file via atomic rename, so a reader never sees a half-written
one. Used by scripts/generate_benchmark_vllm.py (per finished row) and scripts/grade_benchmark_
outputs.py (every 25 scored rows); scripts/train_sft.py keeps its own per-step status.json, which
already serves the same purpose at a coarser (step, not row) granularity and was not touched here
to avoid re-risking the training path right after its multi-GPU rewrite.
"""
import json
import time
from pathlib import Path


def write(path, **fields):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps({"updated": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields}, indent=1))
    tmp.replace(path)
