"""Evaluate a fixed list of models on one benchmark protocol, each model `repeats` times.

Suite json (configs/eval_suites/*.json):
  {"name", "protocol_id", "data", "data_sha256", "max_new_tokens", "gpus": [...], "repeats": {name: k},
   "models": [{"name", "model_version" + "checkpoint"} | {"name", "model"}]}
A model_version entry is a LoRA checkpoint (scripts/evaluate_model_version.py); a "model" entry is a
full model directory (base or merged), generated directly with vLLM. Repeats re-run the identical
setting; the spread across them is the evaluation noise (vLLM batching is not bit-deterministic).
Outputs: results/eval-suites/<name>/<model>/rep<k>/ (generated.jsonl, cases.jsonl, summary.json);
for checkpoints that path is a symlink to <run>/evaluations/<checkpoint>/<protocol>/rep<k>, since
evaluation output must stay inside its model-version run. Finished repetitions are skipped, so
the suite can be re-run after an interruption.

  python scripts/run_eval_suite.py configs/eval_suites/v2-scale256.json
"""
import json, os, queue, subprocess, sys, threading, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mopd.runtime.provenance import sha256
from run_experiment import EVAL_SCHEDULING, PY


def target(suite, m, out):
    """Where the evaluation really writes: inside the run for checkpoints, else the suite path."""
    if "model_version" in m:
        return Path(m["model_version"]).parent / "evaluations" / m["checkpoint"] / suite["protocol_id"] / out.name
    return out


def commands(suite, m, out, gpu):
    common = ["--data", suite["data"], "--protocol-id", suite["protocol_id"], "--max-new-tokens", str(suite["max_new_tokens"])]
    if "model_version" in m:
        return [[PY, "scripts/evaluate_model_version.py", "--model-version", m["model_version"], "--checkpoint", m["checkpoint"],
                 "--out", str(out), "--device", str(gpu), *common, *EVAL_SCHEDULING]]
    return [[str(ROOT / "mopd_infer/bin/python"), "scripts/generate_benchmark_vllm.py", "--model", m["model"], "--out", str(out),
             "--device", str(gpu), "--model-version-id", m.get("model_version_id", f"model:{Path(m['model']).name}"),
             "--student-prompt-format", "plain_qa_v1", *common, *EVAL_SCHEDULING],
            [PY, "scripts/grade_benchmark_outputs.py", "--data", suite["data"], "--out", str(out)]]


def main():
    suite = json.loads(Path(sys.argv[1]).read_text())
    if sha256(suite["data"]) != suite["data_sha256"]:
        raise SystemExit("benchmark file digest differs from the suite pin")
    root = ROOT / "results/eval-suites" / suite["name"]
    jobs = queue.Queue()
    for m in suite["models"]:
        for k in range(1, suite.get("repeats", {}).get(m["name"], 1) + 1):
            out = root / m["name"] / f"rep{k}"
            if not (out / "summary.json").exists():
                jobs.put((m, out))
    print(f"{jobs.qsize()} evaluations to run on GPUs {suite['gpus']}", flush=True)
    failed = []

    def worker(gpu):
        while True:
            try:
                m, out = jobs.get_nowait()
            except queue.Empty:
                return
            out.parent.mkdir(parents=True, exist_ok=True)
            real = target(suite, m, out)
            if real != out and not out.is_symlink():
                real.parent.mkdir(parents=True, exist_ok=True)
                out.symlink_to(real, target_is_directory=True)
            start = time.time()
            with open(str(out) + ".log", "a") as log:
                for cmd in commands(suite, m, real, gpu):
                    if subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode:
                        failed.append(str(out)); break
            print(json.dumps({"done": str(out.relative_to(root)), "gpu": gpu, "seconds": round(time.time() - start),
                              "ok": str(out) not in failed}), flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in suite["gpus"]]
    for t in threads: t.start()
    for t in threads: t.join()
    if failed:
        raise SystemExit(f"failed: {failed}")


if __name__ == "__main__":
    main()
