"""Execute an experiment's job graph (configs/experiments/<name>/plan.json) on the server.

Before anything starts: the deployed source must be a clean, unmodified commit, the spec
sha256 must match plan.json, every data pin must verify (mopd.runtime.provenance), and the
experiment's GPUs must be idle. Then jobs start as soon as their dependencies finish:
  Jobs run through mopd.runtime.scheduler: train_mopd and full-parameter train_sft take their fixed
  GPU set (after a collective-bandwidth probe), evals take a fair share of the idle GPUs, merges one
  GPU, exports the CPU.
A job already finished (run complete, summary.json present, .finalized marker) is skipped,
so the same command resumes after an interruption; a half-written training run fails the
job instead of being overwritten. Every job start/finish is logged with free disk space.

  python scripts/run_experiment.py configs/experiments/scale256/plan.json
"""

import argparse, hashlib, json, os, shutil, subprocess, sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mopd.runtime.provenance import _source, _data, sha256
from mopd.evaluation.eval_commands import (
    eval_commands,
    PY,
    SCHEDULING as EVAL_SCHEDULING,
)  # re-exported: run_eval_suite.py imports these
from mopd.runtime.scheduler import Scheduler


def free_gib():
    s = os.statvfs(ROOT)
    return s.f_bfree * s.f_frsize / 2**30


def expert_run_config(expert):
    """The SFT run config an expert came from: a LoRA expert records it when exported
    (expert-source.json); a full-parameter expert is its run's own final checkpoint directory."""
    if (expert / "expert-source.json").is_file():
        return json.loads((expert / "expert-source.json").read_text())["run_config"]
    return json.loads((expert.parent / "config.json").read_text())


def expert_identity(expert):
    """What a merge was built from: the LoRA adapter's sha256, or for a full-parameter expert its run,
    checkpoint and weight-file digests."""
    if (expert / "expert-source.json").is_file():
        return json.loads((expert / "expert-source.json").read_text())["adapter_sha256"]
    return {
        "run": str(expert.parent),
        "checkpoint": expert.name,
        "weights_sha256": {f.name: sha256(f) for f in sorted(expert.glob("*.safetensors"))},
    }


def expert_finished(expert):
    if (expert / "expert-source.json").is_file():
        return (expert / ".finalized").exists()
    status = expert.parent / "status.json"
    return (
        (expert / "config.json").is_file()
        and status.is_file()
        and json.loads(status.read_text()).get("state") == "complete"
    )


class Runner:
    def __init__(self, plan_path, min_free_gib):
        self.plan = json.loads(Path(plan_path).read_text())
        self.min_free = min_free_gib
        name = self.plan["experiment"]["name"]
        self.logdir = ROOT / "logs/experiments" / name
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.jobs = {j["id"]: j for j in self.plan["jobs"]}
        self.state = {}
        self.lock = threading.Lock()
        self.busy = set()

    def log(self, event, **fields):
        record = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "event": event,
            "ssd_free_gib": round(free_gib(), 1),
            **fields,
        }
        with self.lock, (self.logdir / "events.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    def preflight(self):
        exp = self.plan["experiment"]
        if sha256(exp["spec_path"]) != exp["spec_sha256"]:
            raise SystemExit("spec changed since plan.json was generated")
        spec = json.loads(Path(exp["spec_path"]).read_text())
        source = _source(ROOT)
        _data({"data_pins": spec["data_pins"]})
        used = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        ).stdout
        busy = {int(i): int(m) for i, m in (l.split(",") for l in used.strip().splitlines())}
        for name, ref in self.plan.get("reuse_experts", {}).items():
            src = expert_run_config(Path(ref["dir"]))
            if (src.get("experiment") or {}).get("name") != ref["from_experiment"]:
                raise SystemExit(
                    f"reused expert {name}: {ref['dir']} was not trained by experiment {ref['from_experiment']}"
                )
            # micro_batch_size is not compared: it only splits a step's backward passes (gradients are
            # micro-batch-invariant, verify_microbatch_invariance.py), and MOPD may need a smaller one
            # than its experts' SFT to fit long rollouts next to a full-parameter student.
            mapping = {
                "student": "student",
                "seed": "seed",
                "steps": "steps",
                "lr": "lr",
                "lr_schedule": "lr_schedule",
                "sft_max_seq_len": "max_seq_len",
            }
            diff = {
                k: (self.plan["controls"][k], src.get(v))
                for k, v in mapping.items()
                if self.plan["controls"][k] != src.get(v)
            }
            if sum(d["count"] for d in src["domains"].values()) != self.plan["controls"]["samples_per_step"]:
                diff["samples_per_step"] = "differs"
            if diff:
                raise SystemExit(f"reused expert {name} was trained with different controls: {diff}")
            if not expert_finished(Path(ref["dir"])):
                raise SystemExit(f"reused expert {name} is not finished")
        taken = [g for g in self.plan["gpus"] if busy.get(g, 0) > 1000]
        if taken:
            raise SystemExit(f"GPUs not idle: {taken}")
        self.log("preflight_ok", source=source["source_version"], pins=len(spec["data_pins"]))

    def done(self, j):
        k = j["kind"]
        if k in ("train_sft", "train_mopd"):
            status = Path(json.loads((ROOT / j["config"]).read_text())["output"]) / "status.json"
            return status.exists() and json.loads(status.read_text()).get("state") == "complete"
        if k == "eval_checkpoint":
            run = Path(j["model_version"]).parent
            return (
                run / "evaluations" / j["checkpoint"] / self.plan["evaluation"]["protocol_id"] / "summary.json"
            ).exists()
        if k in ("export_expert", "merge"):
            return (Path(j["out"]) / ".finalized").exists()
        if k == "eval_model":
            return (ROOT / j["out"] / "summary.json").exists()
        raise ValueError(k)

    def commands(self, j, gpus):
        ev = self.plan["evaluation"]
        k = j["kind"]
        base = self.plan["student"]
        if k == "train_sft":  # torchrun for any GPU count: one GPU behaves like a plain run, several shard (hf_full)
            return [
                (
                    [
                        PY,
                        "-m",
                        "torch.distributed.run",
                        "--standalone",
                        f"--nproc_per_node={len(gpus)}",
                        "scripts/train_sft.py",
                        "--config",
                        j["config"],
                    ],
                    {"CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus))},
                )
            ]
        if k == "train_mopd":  # a sharded student (nproc ranks on GPUs 0..nproc-1) goes through torchrun
            if "nproc" in j:
                return [
                    (
                        [
                            PY,
                            "-m",
                            "torch.distributed.run",
                            "--standalone",
                            f"--nproc_per_node={j['nproc']}",
                            "scripts/train_multidomain_opd.py",
                            "--config",
                            j["config"],
                        ],
                        {},
                    )
                ]
            return [([PY, "-u", "scripts/train_multidomain_opd.py", "--config", j["config"]], {})]
        if k == "eval_checkpoint":
            return eval_commands(
                data=ev["data"],
                protocol_id=ev["protocol_id"],
                max_new_tokens=ev["max_new_tokens"],
                device=gpus[0],
                model_version=j["model_version"],
                checkpoint=j["checkpoint"],
            )
        if k == "export_expert":
            return [
                (
                    [PY, "scripts/export_lora_expert.py", "--base", base, "--adapter", j["adapter"], "--out", j["out"]],
                    {"CUDA_VISIBLE_DEVICES": ""},
                ),
                ([PY, "scripts/finalize_merged_model.py", "--base", base, j["out"]], {}),
            ]
        if k == "merge":
            return [
                (
                    [str(ROOT / "mopd_merge/bin/mergekit-yaml"), j["yaml"], j["out"], "--cuda", "--lazy-unpickle"],
                    {"CUDA_VISIBLE_DEVICES": str(gpus[0])},
                ),
                ([PY, "scripts/finalize_merged_model.py", "--base", base, j["out"]], {}),
            ]
        if k == "eval_model":
            return eval_commands(
                data=ev["data"],
                protocol_id=ev["protocol_id"],
                max_new_tokens=ev["max_new_tokens"],
                device=gpus[0],
                out=ROOT / j["out"],
                model=j["model"],
                model_version_id=f"merge:{Path(j['model']).name}",
            )
        raise ValueError(k)

    def finish_artifact(self, j):
        out = Path(j["out"])
        if j["kind"] == "merge":
            shutil.copy(ROOT / j["yaml"], out / Path(j["yaml"]).name)
            experts = [
                l.split("model:", 1)[1].strip() for l in (ROOT / j["yaml"]).read_text().splitlines() if "model:" in l
            ]
            (out / "merge-source.json").write_text(
                json.dumps(
                    {
                        "yaml": j["yaml"],
                        "yaml_sha256": sha256(ROOT / j["yaml"]),
                        "experts": {e: expert_identity(Path(e)) for e in experts},
                    },
                    indent=2,
                )
            )
        (out / ".finalized").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))

    def gpu_spec(self, j):
        """Fixed sets (MOPD layouts, sharded SFT) as given; single-GPU jobs become a fair share of the
        idle GPUs (evaluation hands rows to every GPU it gets); merges stay on one GPU."""
        if isinstance(j["gpus"], list):
            return j["gpus"]
        if j["gpus"] == 0:
            return 0
        return {"min": 1, "max": 1} if j["kind"] == "merge" else {"min": 1}

    def run_job(self, j, gpus):
        log = self.logdir / (j["id"].replace(":", "__") + ".log")
        if j["kind"] in ("train_sft", "train_mopd"):
            output = Path(json.loads((ROOT / j["config"]).read_text())["output"])
            if output.exists():
                raise RuntimeError(f"incomplete run directory exists: {output}; inspect before rerunning")
        for cmd, extra in self.commands(j, gpus):
            with log.open("a") as f:
                f.write(f"\n$ {' '.join(cmd)}\n")
                f.flush()
                subprocess.run(
                    cmd, cwd=ROOT, env={**os.environ, **extra}, stdout=f, stderr=subprocess.STDOUT, check=True
                )
        if j["kind"] in ("export_expert", "merge"):
            self.finish_artifact(j)

    def loop(self):
        order = {"train_mopd": 0, "train_sft": 1, "merge": 2, "export_expert": 3, "eval_model": 4, "eval_checkpoint": 5}
        sched = Scheduler(self.plan["gpus"], self.logdir, min_free_gib=self.min_free, name="experiment")
        for j in self.jobs.values():
            sharded = (
                j["kind"] == "train_mopd" and "nproc" in j or j["kind"] == "train_sft" and isinstance(j["gpus"], list)
            )
            sched.add(
                {
                    "id": j["id"],
                    "deps": j["deps"],
                    "gpus": self.gpu_spec(j),
                    "order": order[j["kind"]],
                    "gate": "collectives" if sharded else None,
                    "done": lambda j=j: self.done(j),
                    "run": lambda gpus, j=j: self.run_job(j, gpus),
                }
            )
        self.state = sched.run()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("plan")
    p.add_argument("--min-free-gib", type=float, default=100)
    a = p.parse_args()
    r = Runner(a.plan, a.min_free_gib)
    r.preflight()
    r.loop()


if __name__ == "__main__":
    main()
