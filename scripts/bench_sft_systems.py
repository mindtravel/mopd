"""Time a few train_sft steps of one sweep variant under different training-system settings
(FSDP resharding, gradient checkpointing, micro-batch size), to pick the fastest that fits.

  python scripts/bench_sft_systems.py configs/sweeps/code-full-v1.json oci --steps 6 \
      --setting '{"fsdp": {"reshard_after_forward": true}}' '{"fsdp": {"reshard_after_forward": false}}'

Each setting is a JSON object merged into the variant's SFT config. Step 1 (warm-up) is excluded.
"""
import argparse, json, os, shutil, statistics, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from mopd.data import recipes
from mopd.evaluation.eval_commands import PY
from run_sweep import Sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep"); ap.add_argument("variant")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--setting", nargs="+", required=True)
    a = ap.parse_args()
    s = Sweep(a.sweep)
    v = next(x for x in s.spec["variants"] if x["name"] == a.variant)
    base = s.config(v, recipes.build(ROOT / v["recipe"]))
    scratch = ROOT / "runs/scratch/bench-sft"
    for i, setting in enumerate(a.setting):
        cfg = {**base, **json.loads(setting), "steps": a.steps, "save_every": 10**6, "output": str(scratch / str(i))}
        shutil.rmtree(scratch / str(i), ignore_errors=True)
        (scratch).mkdir(parents=True, exist_ok=True)
        path = scratch / f"{i}.json"; path.write_text(json.dumps(cfg, indent=1))
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, a.gpus))}
        with (scratch / f"{i}.log").open("w") as log:
            rc = subprocess.run([PY, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={len(a.gpus)}",
                                 "scripts/train_sft.py", "--config", str(path)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        metrics = [json.loads(l) for l in (scratch / str(i) / "metrics.jsonl").read_text().splitlines()] if rc == 0 else []
        timed = metrics[1:]
        print(json.dumps({"setting": json.loads(setting), "ok": rc == 0,
            "sec_per_step": round(statistics.median(m["seconds"] for m in timed), 2) if timed else None,
            "tokens_per_sec": round(sum(m["tokens"] for m in timed) / sum(m["seconds"] for m in timed)) if timed else None,
            "max_memory_gib": max((m["max_memory_gib"] for m in metrics), default=None),
            "loss": [round(m["loss"], 4) for m in metrics]}), flush=True)
        shutil.rmtree(scratch / str(i), ignore_errors=True)  # the final full checkpoint is 8 GB


if __name__ == "__main__":
    main()
