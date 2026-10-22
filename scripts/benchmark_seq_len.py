"""Measure SFT step time and peak memory at different sequence-length caps and micro-batch sizes,
using real OCR trace lengths, with the same full-model gradient checkpointing train_sft.py should
use (it currently doesn't -- see the comment below).

Packing is already per-micro-batch (pack() pads to that batch's own longest row, not a fixed
global width), so the cap mainly decides which rows are admitted and how long the occasional long
micro-batch can get. This measures that effect directly instead of estimating it.

  python scripts/benchmark_seq_len.py --caps 4096 8192 16384 20480 --micro 1 2 --device cuda:0
"""
import argparse, glob, random, time
from pathlib import Path
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mopd.backends.student_adapter import attach_student_adapter
from mopd.training.batching import pack, logps
ROOT = Path(__file__).resolve().parents[1]


def load_rows(n):
    rows = []
    for f in sorted(glob.glob(str(ROOT / "data/raw/opencodereasoning/split_0/*.parquet"))):
        rows += pq.ParquetFile(f).read(columns=["input", "output"]).slice(0, n).to_pylist()
        if len(rows) >= n:
            break
    random.Random(0).shuffle(rows)
    return rows[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", type=int, nargs="+", default=[4096, 8192, 16384, 20480])
    ap.add_argument("--micro", type=int, nargs="+", default=[2])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--student", default=str(ROOT / "models/Qwen3-4B-Base"))
    ap.add_argument("--no-checkpointing", action="store_true", help="measure train_sft.py's current (unsafe) behavior")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.student)
    rows = load_rows(600)
    lens, pref, resp = {}, {}, {}
    for r in rows:
        key = id(r)
        pref[key] = tok(r["input"], add_special_tokens=False).input_ids
        resp[key] = tok(r["output"], add_special_tokens=False).input_ids + [tok.eos_token_id]
        lens[key] = len(pref[key]) + len(resp[key])

    base = AutoModelForCausalLM.from_pretrained(a.student, dtype=torch.bfloat16, local_files_only=True).to(a.device)
    if not a.no_checkpointing:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = attach_student_adapter(base, {})
    model.enable_input_require_grads()
    model.train()  # gradient checkpointing only recomputes (instead of retaining) activations in train mode
    params = [p for p in model.parameters() if p.requires_grad]
    pad = tok.pad_token_id or tok.eos_token_id

    for cap in a.caps:
        pool = [r for r in rows if lens[id(r)] <= cap]
        for micro in a.micro:
            if len(pool) < micro * 3:
                print(f"cap={cap} micro={micro}: only {len(pool)} rows fit, skipping", flush=True)
                continue
            batches, longest = [], 0
            for i in range(0, len(pool) - micro + 1, micro):
                group = pool[i:i + micro]
                batches.append(group)
                longest = max(longest, max(lens[id(r)] for r in group))
                if len(batches) == 8:
                    break
            torch.cuda.reset_peak_memory_stats(a.device)
            times = []
            try:
                for group in batches:
                    seqs = [pref[id(r)] + resp[id(r)] for r in group]
                    starts = [len(pref[id(r)]) for r in group]
                    ids, attention, mask = pack(seqs, starts, pad, a.device)
                    for p in params:
                        p.grad = None
                    torch.cuda.synchronize()
                    t0 = time.time()
                    losses = logps(model, ids, attention, mask)
                    loss = -sum(l.mean() for l in losses) / len(losses)
                    loss.backward()
                    torch.cuda.synchronize()
                    times.append(time.time() - t0)
            except torch.OutOfMemoryError:
                print(f"cap={cap:6d} micro={micro}  OOM at batch {len(times) + 1}/{len(batches)} "
                      f"(widest batch attempted so far, up to longest_batch_width={longest})", flush=True)
                for p in params:
                    p.grad = None
                torch.cuda.empty_cache()
                continue
            if not times:
                continue
            mean = sum(times[1:]) / len(times[1:]) if len(times) > 1 else times[0]
            peak = torch.cuda.max_memory_allocated(a.device) / 2**30
            print(f"cap={cap:6d} micro={micro}  pool={len(pool)}/{len(rows)} ({100 * len(pool) / len(rows):.0f}%)  "
                  f"batches={len(batches)}  longest_batch_width={longest}  "
                  f"mean_step={mean * 1000:.0f}ms  peak_mem={peak:.1f}GiB", flush=True)


if __name__ == "__main__":
    main()
