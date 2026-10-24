"""Evaluate fixed heldout prompts at Base and saved checkpoints during training."""

import json, os, subprocess, time, traceback
from pathlib import Path

ROOT = Path("/ssd/lzx/mopd")
RUN = ROOT / "runs/20261013-dual-domain-batch32-mgpu-opd"
OUT = RUN / "validation/ability"
OUT.mkdir(parents=True, exist_ok=True)


def status(**data):
    p = OUT / "status.partial"
    p.write_text(json.dumps({"time": time.time(), **data}, indent=2))
    p.replace(OUT / "status.json")


def evaluate(step):
    target = OUT / f"step-{step:04d}"
    if (target / "summary.json").exists():
        return
    if target.exists():
        raise RuntimeError("Incomplete evaluation directory; inspect before retry: " + str(target))
    args = [
        str(ROOT / "mopd_train/bin/python"),
        "-u",
        str(ROOT / "scripts/eval_student_batches.py"),
        "--model",
        str(ROOT / "models/Qwen3-4B-Base"),
        "--data",
        str(ROOT / "data/dual_eval/pilot.jsonl"),
        "--out",
        str(target),
        "--device",
        "cuda:3",
        "--batch",
        "4",
    ]
    if step:
        args += ["--adapter", str(RUN / f"checkpoint-{step:04d}")]
    status(state="evaluating", step=step, device="cuda:3")
    with (ROOT / "logs" / f"dual-ability-{step:04d}.log").open("a") as f:
        p = subprocess.Popen(args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        status(state="evaluating", step=step, device="cuda:3", pid=p.pid)
        if p.wait() != 0:
            raise RuntimeError("Evaluation failed at step " + str(step))


def main():
    try:
        evaluate(0)
        for step in [16, 32, 48, 64]:
            while not (RUN / f"checkpoint-{step:04d}/training_state.pt").exists():
                st = json.loads((RUN / "status.json").read_text())
                if st.get("state") == "failed":
                    raise RuntimeError("Training failed")
                status(state="waiting_checkpoint", step=step)
                time.sleep(30)
            evaluate(step)
        status(state="complete")
    except BaseException:
        status(state="failed", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
