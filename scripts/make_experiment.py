"""Generate every arm's config of one comparison from a single experiment spec, and check them.

The spec (configs/experiments/<name>.json) is the only place a controlled variable is
written down: student, seed, steps, samples per step, lr and schedule, micro batch,
checkpoint cadence, data pools with pinned sha256, evaluation protocol. This script turns
it into one config per arm under configs/experiments/<name>/ plus plan.json (the job
graph scripts/run_experiment.py executes). Nothing is read from the data or the server,
so the output is a pure function of the spec and is committed to git like code.

  python scripts/make_experiment.py configs/experiments/scale256.json          # write
  python scripts/make_experiment.py configs/experiments/scale256.json --check  # verify

--check regenerates in memory and requires (a) every committed file to be byte-identical
to what the spec produces, so a hand-edited config is caught, and (b) the controlled
fields to be identical across all training arms. At run time mopd.runtime.provenance
re-verifies the spec sha256 and every data pin before the first step.

"reuse_experts" lets a spec take finished experts from an earlier experiment instead of
retraining them; scripts/run_experiment.py refuses to start unless each reused expert's
source run belongs to the named experiment and was trained with this spec's controls.

Mixed SFT and MOPD arms use the same domain order, quotas and seed, and both entries
plan batches with mopd.data.mixture.plan_domains, so they see the same prompts at every step.
"""

from mopd import paths
from mopd.training import controls as training_controls
import argparse, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = str(paths.ROOT)


def dumps(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def domains_spec(spec, domains):
    counts = spec["quotas"][str(len(domains))]
    return {d: {**spec["pools"][d], "count": n, "weight": 1 / len(domains)} for d, n in zip(domains, counts)}


def ordered(spec, domains):
    return [d for d in spec["domain_order"] if d in domains]


def generate(spec_path):
    spec_bytes = Path(spec_path).read_bytes()
    spec = json.loads(spec_bytes)
    training_controls.check(spec["controls"], spec_path)
    c = spec["controls"]
    rel = Path(spec_path).resolve().relative_to(ROOT)
    experiment = lambda arm: {
        "name": spec["name"],
        "arm": arm,
        "spec_path": f"{REMOTE_ROOT}/{rel}",
        "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
    }
    arms = {a["name"]: a for a in spec["arms"]}
    reuse = spec.get("reuse_experts", {})  # name -> {"dir", "from_experiment"}: trained by an earlier spec
    # Full fine-tuning (student_backend hf_full, mopd.training.fsdp): every SFT run spans all of the
    # experiment's GPUs, and its final checkpoint already is a standalone model -- it is the expert
    # directly, with no LoRA export step.
    full = c["student_backend"] == "hf_full"
    run_dir = lambda name: spec["output_prefix"] + name
    final = f"checkpoint-{c['steps']:04d}"
    expert_dir = lambda name: (
        reuse[name]["dir"]
        if name in reuse
        else f"{run_dir(name)}/{final}" if full else spec["expert_dir_prefix"] + name
    )
    expert_dep = lambda name: [] if name in reuse else [f"train:{name}"] if full else [f"export:{name}"]
    files, jobs = {}, []
    ev = spec["evaluation"]

    def eval_jobs(name, deps):
        for step in ev["checkpoints"]:
            jobs.append(
                {
                    "id": f"eval:{name}:{step}",
                    "kind": "eval_checkpoint",
                    "gpus": 1,
                    "deps": deps,
                    "model_version": f"{run_dir(name)}/model-version.json",
                    "checkpoint": f"checkpoint-{step:04d}",
                }
            )

    for name, arm in arms.items():
        common = {
            **training_controls.training_fields(
                c,
                micro_batch_size=c["micro_batch_size"],
                method=arm["method"] if arm["method"] in ("sft", "mopd") else None,
                where=f"{spec_path}: arm {name}",
            ),
            "output": run_dir(name),
            "data_pins": spec["data_pins"],
            "experiment": experiment(name),
            **training_controls.system_fields(c),
        }
        if arm["method"] == "sft":
            cfg = {
                **common,
                "training_method": "sft",
                "student_device": "cuda:0",
                "max_seq_len": c["sft_max_seq_len"],
                "domains": domains_spec(spec, ordered(spec, arm["domains"])),
            }
            files[f"{name}.json"] = dumps(cfg)
            jobs.append(
                {
                    "id": f"train:{name}",
                    "kind": "train_sft",
                    "gpus": list(spec["gpus"]) if full else 1,
                    "deps": [],
                    "config": f"configs/experiments/{spec['name']}/{name}.json",
                }
            )
            eval_jobs(name, [f"train:{name}"])
            if not full and any(
                name in a.get("experts", {}).values() or name in a.get("teachers", {}).values() for a in arms.values()
            ):
                jobs.append(
                    {
                        "id": f"export:{name}",
                        "kind": "export_expert",
                        "gpus": 0,
                        "deps": [f"train:{name}"],
                        "adapter": f"{run_dir(name)}/{final}",
                        "out": expert_dir(name),
                    }
                )
        elif arm["method"] == "mopd":
            doms = ordered(spec, arm["teachers"])
            dev = arm["devices"]
            # full: the student is one torchrun rank per GPU, devices["student"] = [0..n-1]
            students = dev["student"] if full else [dev["student"]]
            if full and students != list(range(len(students))):
                raise ValueError(f"{name}: a sharded student runs on GPUs 0..n-1")
            if len(dev["teachers"]) != len(doms):
                raise ValueError(f"{name}: one teacher device per domain")
            dspec = domains_spec(spec, doms)
            for d, gpu in zip(doms, dev["teachers"]):
                dspec[d] = {
                    "teacher": expert_dir(arm["teachers"][d]),
                    "device": f"cuda:{gpu}",
                    "teacher_prompt_format": c["teacher_prompt_format"],
                    **dspec[d],
                }
            cfg = {
                **common,
                "training_method": "mopd",
                "student_device": f"cuda:{students[0]}",
                "domains": dspec,
                **({"student_gpus": len(students)} if full else {}),
                "max_new_tokens": c["mopd_max_new_tokens"],
                "generation_batch_size": c["mopd_generation_batch_size"],
                "rollout_backend": "vllm",
                "rollout_devices": [f"cuda:{g}" for g in dev["rollout"]],
                "vllm_memory_utilization": c["mopd_vllm_memory_utilization"],
            }
            files[f"{name}.json"] = dumps(cfg)
            gpus = sorted({*students, *dev["teachers"], *dev["rollout"]})
            jobs.append(
                {
                    "id": f"train:{name}",
                    "kind": "train_mopd",
                    "gpus": gpus,
                    "deps": [d for t in arm["teachers"].values() for d in expert_dep(t)],
                    "config": f"configs/experiments/{spec['name']}/{name}.json",
                    **({"nproc": len(students)} if full else {}),
                }
            )
            eval_jobs(name, [f"train:{name}"])
        elif arm["method"] == "merge_linear":
            doms = ordered(spec, arm["experts"])
            lines = [
                f"# experiment {spec['name']} / {name}; spec sha256 {experiment(name)['spec_sha256']}",
                "merge_method: linear",
                "models:",
            ]
            for d in doms:
                lines += [f"  - model: {expert_dir(arm['experts'][d])}", "    parameters:", "      weight: 1.0"]
            lines += ["parameters:", "  normalize: true", "dtype: bfloat16"]
            files[f"{name}.yaml"] = "\n".join(lines) + "\n"
            merged = spec["merge_dir_prefix"] + name
            jobs.append(
                {
                    "id": f"merge:{name}",
                    "kind": "merge",
                    "gpus": 1,
                    "deps": [x for d in doms for x in expert_dep(arm["experts"][d])],
                    "yaml": f"configs/experiments/{spec['name']}/{name}.yaml",
                    "out": merged,
                }
            )
            jobs.append(
                {
                    "id": f"eval:{name}",
                    "kind": "eval_model",
                    "gpus": 1,
                    "deps": [f"merge:{name}"],
                    "model": merged,
                    "out": f"results/experiments/{spec['name']}/{name}/{ev['protocol_id']}",
                }
            )
        else:
            raise ValueError(f"{name}: unknown method {arm['method']!r}")
    controls = {
        k: (training_controls.lr_for(c, "sft", spec_path) if k == "lr" else c[k])
        for k in (
            "student",
            "seed",
            "steps",
            "samples_per_step",
            "lr",
            "lr_schedule",
            "micro_batch_size",
            "sft_max_seq_len",
        )
    }
    plan = {
        "experiment": experiment(None),
        "evaluation": ev,
        "gpus": spec["gpus"],
        "student": c["student"],
        **({"reuse_experts": reuse, "controls": controls} if reuse else {}),
        "jobs": jobs,
    }  # older plans stay byte-identical
    files["plan.json"] = dumps(plan)
    return spec, files


CONTROLLED = (
    "student",
    "student_backend",
    "student_prompt_format",
    "seed",
    "steps",
    "lr",
    "lr_schedule",
    "micro_batch_size",
    "save_every",
    "data_pins",
)


def check_controls(spec, files):
    configs = {n: json.loads(t) for n, t in files.items() if n.endswith(".json") and n != "plan.json"}
    reference = next(iter(configs.values()))
    # lr is controlled within a training method (SFT vs MOPD have their own defaults: LR_DEFAULTS)
    lr_by_method = {}
    for name, cfg in configs.items():
        for key in CONTROLLED:
            if key == "lr":
                if lr_by_method.setdefault(cfg.get("training_method"), cfg["lr"]) != cfg["lr"]:
                    raise ValueError(f"{name}: lr differs from the other {cfg.get('training_method')} arms")
            elif cfg[key] != reference[key]:
                raise ValueError(f"{name}: controlled field {key!r} differs")
        if sum(d["count"] for d in cfg["domains"].values()) != spec["controls"]["samples_per_step"]:
            raise ValueError(f"{name}: samples per step != {spec['controls']['samples_per_step']}")
        for d, ds in cfg["domains"].items():
            if (
                ds["train_file"] != spec["pools"][d]["train_file"]
                or ds["holdout_files"] != spec["pools"][d]["holdout_files"]
            ):
                raise ValueError(f"{name}: domain {d} pool differs from spec")
            for path in [ds["train_file"], *ds["holdout_files"]]:
                if path not in spec["data_pins"]:
                    raise ValueError(f"{name}: {path} is not pinned")
    return sorted(configs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("spec")
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    spec, files = generate(a.spec)
    out = ROOT / "configs/experiments" / spec["name"]
    arms = check_controls(spec, files)
    if a.check:
        on_disk = {f.name for f in out.iterdir()} if out.exists() else set()
        stale = [n for n in files if not (out / n).exists() or (out / n).read_text() != files[n]]
        extra = sorted(on_disk - set(files))
        if stale or extra:
            sys.exit(f"configs differ from spec: changed/missing {stale}, unexpected {extra}")
        print(f"ok: {len(files)} files match spec; controlled fields identical across {len(arms)} training arms")
        return
    out.mkdir(parents=True, exist_ok=True)
    for n, t in files.items():
        (out / n).write_text(t)
    print(f"wrote {len(files)} files to {out.relative_to(ROOT)}: {sorted(files)}")


if __name__ == "__main__":
    main()
