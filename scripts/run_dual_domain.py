"""Download-gated smoke, then 64-step multi-domain training and paired pilot eval."""

import json, os, subprocess, time, traceback
from pathlib import Path

ROOT = Path("/ssd/lzx/mopd")
STATUS = ROOT / "logs/dual-domain-supervisor-status.json"


def status(**kwargs):
    p = STATUS.with_suffix(".partial")
    p.write_text(json.dumps({"time": time.time(), **kwargs}, indent=2))
    p.replace(STATUS)


def job(script, args, log):
    status(stage=script, log=log)
    with (ROOT / "logs" / log).open("a") as f:
        p = subprocess.Popen(
            [str(ROOT / "mopd_train/bin/python"), "-u", str(ROOT / "scripts" / script), *args],
            cwd=ROOT,
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        status(stage=script, pid=p.pid, log=log)
        if p.wait() != 0:
            raise RuntimeError(script + " failed; see " + log)


def main():
    try:
        deadline = time.time() + 7200
        while not (ROOT / "models/Qwen3-4B-Instruct-2507/verified.json").exists():
            status(stage="waiting_for_verified_teacher")
            if time.time() > deadline:
                raise TimeoutError("Teacher download did not finish")
            time.sleep(30)
        job(
            "train_multidomain_opd.py",
            ["--config", str(ROOT / "configs/dual_domain_smoke.json")],
            "dual-domain-smoke.log",
        )
        smoke = json.loads((ROOT / "runs/20261013-dual-domain-batch32-mgpu-smoke/status.json").read_text())
        if (
            smoke.get("state") != "complete"
            or not all(smoke.get("teachers_unchanged", {}).values())
            or smoke.get("adapter_delta_l2", 0) <= 0
        ):
            raise RuntimeError("Smoke verification failed")
        job("train_multidomain_opd.py", ["--config", str(ROOT / "configs/dual_domain.json")], "dual-domain-train.log")
        run = ROOT / "runs/20261013-dual-domain-batch32-mgpu-opd"
        result = json.loads((run / "status.json").read_text())
        if result.get("state") != "complete":
            raise RuntimeError("Training incomplete")
        out = ROOT / "runs/20261013-dual-domain-evaluation"
        common = [
            "--data",
            str(ROOT / "data/dual_eval/pilot.jsonl"),
            "--device",
            "cuda:0",
            "--max-new-tokens",
            "8192",
            "--model",
            str(ROOT / "models/Qwen3-4B-Base"),
        ]
        job("eval_domains.py", [*common, "--out", str(out / "student_before")], "dual-eval-before.log")
        job(
            "eval_domains.py",
            [*common, "--out", str(out / "student_after"), "--adapter", result["final_adapter"]],
            "dual-eval-after.log",
        )
        job(
            "eval_domains.py",
            [
                "--data",
                str(ROOT / "data/dual_eval/instruction.jsonl"),
                "--device",
                "cuda:0",
                "--max-new-tokens",
                "8192",
                "--model",
                str(ROOT / "models/Qwen3-4B-Instruct-2507"),
                "--teacher",
                "--out",
                str(out / "instruction_teacher"),
            ],
            "dual-eval-teacher.log",
        )
        before = json.loads((out / "student_before/summary.json").read_text())
        after = json.loads((out / "student_after/summary.json").read_text())
        (out / "comparison.json").write_text(json.dumps({"before": before, "after": after}, indent=2))
        status(stage="complete", evaluation=str(out))
    except BaseException:
        status(stage="failed", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
