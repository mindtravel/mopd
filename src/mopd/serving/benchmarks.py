"""The two benchmarks, run from the mopd-bench docker image (docker/benchmarks) against the job's
engine with the fixed protocol (serving.protocol). Each returns its score summary."""

import json
import subprocess

from mopd.serving import protocol

IMAGE = "mopd-bench:latest"


def docker(job_dir, args, log):
    """Host network: the engine listens on the host's 127.0.0.1 (and bridge networking is broken here)."""
    cmd = ["docker", "run", "--rm", "--network", "host", "-e", "API_KEY=EMPTY", "-e", "OPENAI_API_KEY=EMPTY",
           "-v", f"{job_dir}:/out", "--name", f"mopd-bench-{job_dir.name}-{args[0]}", IMAGE, *args]
    with log.open("a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        if subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode != 0:
            raise RuntimeError(f"{args[0]} {args[1] if len(args) > 1 else ''} failed; see {log}")


def ifbench(j, job_dir, api, served):
    """Prompt / instruction accuracy (%), strict and loose."""
    p = protocol.IFBENCH
    log = job_dir / "ifbench.log"
    gen = ["ifbench", "generate", "--run-name", "run", "--model", served, "--api-base", api, "--workers", str(p["workers"]),
           "--max-tokens", str(p["max_tokens"]), "--temperature", str(p["temperature"])]
    docker(job_dir, gen + (["--limit", str(j["ifbench_limit"])] if j.get("ifbench_limit") else []), log)
    docker(job_dir, ["ifbench", "score", "--run-name", "run"], log)
    m = json.loads((job_dir / "ifbench/run/metrics.json").read_text())
    return {mode: {k: round(100 * v, 2) for k, v in m[mode].items() if "accuracy" in k} for mode in ("strict", "loose")}


def automationbench(j, job_dir, api, served):
    """Pass rate (every assertion of a task passes) and mean partial score (%), overall and per domain."""
    p = protocol.AUTOMATIONBENCH
    args = ["automationbench", "--model", served, "--base-url", api, "--api", "chat_completions", "--toolset", p["toolset"],
            "--domains", j["auto_domains"], "--max-steps", str(p["max_steps"]), "--max-concurrent", str(p["max_concurrent"]),
            "--export-json", "/out/automationbench/results.json"]
    docker(job_dir, args + (["--num-examples", str(j["auto_limit"])] if j.get("auto_limit") else []), job_dir / "automationbench.log")
    r = json.loads((job_dir / "automationbench/results.json").read_text())
    pct = lambda v: round(100 * float(v), 2)
    by = {}
    for t in r["tasks"]:
        by.setdefault(str(t["name"]).split(".")[0], []).append(t)
    out = {"all": {"tasks": len(r["tasks"]), "pass_rate": pct(r["summary"]["pass_rate"]), "mean_score": pct(r["summary"]["avg_score"]),
                   "errors": r["summary"].get("tasks_with_errors"), "aborted": len(r["summary"].get("aborted_tasks") or [])}}
    if len(by) > 1:
        for d, ts in by.items():
            out[d] = {"tasks": len(ts), "pass_rate": pct(sum(str(t["passed"]) == "True" for t in ts) / len(ts)),
                      "mean_score": pct(sum(float(t["score"]) for t in ts) / len(ts))}
    return out


RUNNERS = {"ifbench": ifbench, "automationbench": automationbench}
