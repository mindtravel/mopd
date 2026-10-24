"""Evaluate a fixed list of models on one benchmark protocol, each model `repeats` times.

Suite json (configs/eval_suites/*.json):
  {"name", "protocol_id", "data", "data_sha256", "max_new_tokens", "gpus": [...], "repeats": {name: k},
   "models": [{"name", "model_version" + "checkpoint"} | {"name", "model"}]}
A model_version entry is a training run's checkpoint (scripts/evaluate_model_version.py); a "model"
entry is a full model directory (base or merged), generated directly with vLLM. Repeats re-run the
identical setting; the spread across them is the evaluation noise (vLLM batching is not
bit-deterministic). Outputs: results/eval-suites/<name>/<model>/rep<k>/ (generated.jsonl, cases.jsonl,
summary.json); for checkpoints that path is a symlink to <run>/evaluations/<checkpoint>/<protocol>/rep<k>,
since evaluation output must stay inside its model-version run. Jobs run through
mopd.runtime.scheduler (generation on a fair share of the GPUs, grading on the CPU); finished
repetitions are skipped, so the suite can be re-run after an interruption.

  python scripts/run_eval_suite.py configs/eval_suites/v2-scale256.json
"""

import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mopd.evaluation import jobs
from mopd.runtime.gpus import idle_gpus
from mopd.runtime.provenance import sha256
from mopd.runtime.scheduler import Scheduler


def main():
    suite = json.loads(Path(sys.argv[1]).read_text())
    if sha256(suite["data"]) != suite["data_sha256"]:
        raise SystemExit("benchmark file digest differs from the suite pin")
    root = ROOT / "results/eval-suites" / suite["name"]
    sched = Scheduler(idle_gpus(suite["gpus"]), root, name="eval_suite")
    for m in suite["models"]:
        for k in range(1, suite.get("repeats", {}).get(m["name"], 1) + 1):
            link = root / m["name"] / f"rep{k}"
            if "model_version" in m:  # results live inside the run; the suite path links there
                out = jobs.checkpoint_eval_dir(m["model_version"], m["checkpoint"], suite["protocol_id"]) / link.name
                model = dict(model_version=m["model_version"], checkpoint=m["checkpoint"])
                link.parent.mkdir(parents=True, exist_ok=True)
                if not link.is_symlink():
                    out.parent.mkdir(parents=True, exist_ok=True)
                    link.symlink_to(out, target_is_directory=True)
            else:
                out = link
                model = dict(model=m["model"], model_version_id=m.get("model_version_id"))
            for job in jobs.eval_jobs(
                f"{m['name']}:rep{k}",
                out=out,
                data=suite["data"],
                protocol_id=suite["protocol_id"],
                max_new_tokens=suite["max_new_tokens"],
                log=str(link) + ".log",
                **model,
            ):
                sched.add(job)
    state = sched.run()
    failed = sorted(k for k, s in state.items() if s["state"] in ("failed", "blocked"))
    if failed:
        raise SystemExit(f"failed: {failed}")


if __name__ == "__main__":
    main()
