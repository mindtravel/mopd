"""Run a study on the scheduler: one training job per run whose identity has not finished anywhere
under runs/ (role "train"), and every applicable evaluation of every checkpoint (role "eval").
Configs and logs go to results/studies/<study>/ -- derived, so not committed; each run directory
records its full config (and provenance) when training starts."""

import hashlib
import json
from pathlib import Path

from mopd import paths
from mopd.evaluation import jobs
from mopd.runtime.scheduler import Scheduler, usable_gpus
from mopd.study import resolve, spec


def finished_runs():
    """identity -> run directory, for every completed SFT run under runs/ (any study or experiment)."""
    out = {}
    for cfg_path in [*paths.ROOT.glob("runs/*/config.json"), *paths.ROOT.glob("runs/studies/*/*/config.json"),
                     *paths.ROOT.glob("runs/sweeps/*/*/config.json")]:
        run = cfg_path.parent
        try:
            cfg = json.loads(cfg_path.read_text())
            done = json.loads((run / "status.json").read_text()).get("state") == "complete"
        except (OSError, ValueError):
            continue
        if done and cfg.get("training_method") in ("sft", "mopd") and "domains" in cfg and "data_pins" in cfg:
            out.setdefault(resolve.identity(cfg), run)
    return out


def teacher_dirs(study, done):
    """{domain: checkpoint directory} of a MOPD study's teachers: each names a run of another study,
    found wherever its identity finished (so a reused older run is a valid teacher)."""
    out = {}
    for d, ref in (study.get("teachers") or {}).items():
        src_path = resolve.ROOT_CODE / "configs/studies" / f"{ref['study']}.yaml"
        src = spec.load(src_path)
        runs = dict(spec.expand(src))
        if ref["run"] not in runs:
            raise ValueError(f"teacher {d}: no run {ref['run']!r} in study {ref['study']}")
        f = runs[ref["run"]]
        ident = resolve.identity(resolve.run_config(src, ref["run"], f, spec.evals_for(src, f), "", "", resolve.registry()))
        if ident not in done:
            raise ValueError(f"teacher {d}: {ref['study']}/{ref['run']} ({ident}) has not finished training")
        out[d] = Path(done[ident]) / f"checkpoint-{ref['checkpoint']:04d}"
    return out


def plan(study_path):
    """[{name, fields, identity, config, run_dir, reused, evals}] for every run of the study."""
    study = spec.load(study_path)
    reg = resolve.registry()
    rel = Path(study_path).resolve().relative_to(resolve.ROOT_CODE)
    spec_path, spec_sha = f"{paths.ROOT}/{rel}", hashlib.sha256(Path(study_path).read_bytes()).hexdigest()
    done = finished_runs() if paths.ROOT.exists() else {}
    teachers = teacher_dirs(study, done) if study.get("teachers") else None
    runs = []
    for name, fields in spec.expand(study):
        evals = spec.evals_for(study, fields)
        cfg = resolve.run_config(study, name, fields, evals, spec_path, spec_sha, reg, teachers)
        ident = resolve.identity(cfg)
        reused = done.get(ident)
        runs.append({"name": name, "fields": fields, "identity": ident, "config": cfg, "evals": evals,
                     "run_dir": Path(reused) if reused else Path(cfg["output"]), "reused": bool(reused)})
    return study, reg, runs


def run(study_path):
    study, reg, runs = plan(study_path)
    out = paths.ROOT / "results/studies" / study["study"]
    (out / "configs").mkdir(parents=True, exist_ok=True)
    sched = Scheduler(usable_gpus(list(paths.SCHEDULED_GPUS)), out, name=f"study {study['study']}")
    for r in runs:
        name, run_dir = r["name"], r["run_dir"]
        log = out / "logs" / f"{name}.log"
        deps = []
        if not r["reused"]:
            cfg_path = out / "configs" / f"{name}.json"
            cfg_path.write_text(json.dumps(r["config"], indent=2, ensure_ascii=False) + "\n")
            status = run_dir / "status.json"

            cfg = r["config"]
            mopd = cfg["training_method"] == "mopd"

            def train(gpus, cfg_path=cfg_path, run_dir=run_dir, log=log, cfg=cfg, mopd=mopd):
                if run_dir.exists() and not _complete(run_dir):
                    raise RuntimeError(f"incomplete run directory {run_dir}: inspect (or move aside) before retraining")
                # MOPD addresses its layout by physical GPU id (student ranks 0..n-1, teachers, rollout)
                nproc = cfg["student_gpus"] if mopd else len(gpus)
                script = "scripts/train_multidomain_opd.py" if mopd else "scripts/train_sft.py"
                env = {} if mopd else {"CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus))}
                jobs.shell([([str(paths.TRAIN_PY), "-m", "torch.distributed.run", "--standalone",
                              f"--nproc_per_node={nproc}", script, "--config", str(cfg_path)], env)], log)

            if mopd:
                ids = lambda ds: [int(x.split(":")[-1]) for x in ds]
                layout = sorted({*range(cfg["student_gpus"]), *ids(s["device"] for s in cfg["domains"].values()),
                                 *ids(cfg["rollout_devices"])})
                resources = {"gpus": layout, "gate": "collectives"}
            else:
                resources = jobs.training_resources(cfg)
            sched.add({"id": f"train:{name}", **resources, "done": lambda run_dir=run_dir: _complete(run_dir), "run": train})
            deps = [f"train:{name}"]
        for suite in r["evals"]:
            e = reg["evals"][suite]
            for step in r["fields"].get("checkpoints") or study.get("checkpoints") or [r["config"]["steps"]]:
                ck = f"checkpoint-{step:04d}"
                for job in jobs.eval_jobs(f"eval:{name}:{suite}:{step}", out=run_dir / "evaluations" / ck / e["protocol_id"],
                                          data=resolve.abs_path(e["data"]), protocol_id=e["protocol_id"],
                                          max_new_tokens=e["max_new_tokens"], log=log, deps=deps,
                                          model_version=str(run_dir / "model-version.json"), checkpoint=ck):
                    sched.add(job)
    return sched.run()  # final state by job id


def _complete(run_dir):
    try:
        return json.loads((Path(run_dir) / "status.json").read_text()).get("state") == "complete"
    except (OSError, ValueError):
        return False
