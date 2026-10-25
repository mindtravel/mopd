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
    """{domain: checkpoint directory} of a MOPD study's teachers (resolve_refs)."""
    return resolve_refs(study.get("teachers") or {}, done)


def resolve_refs(refs, done):
    """{domain: checkpoint directory} for {domain: {study, run, checkpoint}}: each names a run of a
    study, found wherever its identity finished (so a reused older run is a valid reference)."""
    out = {}
    for d, ref in refs.items():
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
        if fields.get("method") == "merge_linear":  # no training: a weighted average of finished experts
            out = paths.ROOT / "models/merged/studies" / study["study"] / name
            runs.append({"name": name, "fields": fields, "identity": None, "config": {"steps": None}, "evals": evals,
                         "run_dir": out, "reused": False, "merge": resolve_refs(fields["experts"], done)})
            continue
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
        if "merge" in r:
            deps = [_merge_job(sched, out, study, name, r["merge"], run_dir, log)]
            for suite in r["evals"]:
                e = reg["evals"][suite]
                for job in jobs.eval_jobs(f"eval:{name}:{suite}", out=run_dir / "evaluations" / e["protocol_id"],
                                          data=resolve.abs_path(e["data"]), protocol_id=e["protocol_id"],
                                          max_new_tokens=e["max_new_tokens"], log=log, deps=deps, model=str(run_dir),
                                          model_version_id=f"merge:{study['study']}/{name}"):
                    sched.add(job)
            continue
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


def _merge_job(sched, out, study, name, experts, merged, log):
    """mergekit linear merge (normalized equal weights, bf16) of the experts into `merged`, finalized
    against the base model; merge-source.json records the yaml and the expert directories."""
    yaml_path = out / "configs" / f"{name}.yaml"
    lines = [f"# study {study['study']} / {name}", "merge_method: linear", "models:"]
    for d in sorted(experts):
        lines += [f"  - model: {experts[d]}", "    parameters:", "      weight: 1.0"]
    yaml_path.write_text("\n".join(lines + ["parameters:", "  normalize: true", "dtype: bfloat16"]) + "\n")
    base = resolve.abs_path(resolve.registry()["models"][study["base"].get("student", "qwen3-4b-base")]["path"])

    def run(gpus):
        jobs.shell([([str(paths.MERGEKIT), str(yaml_path), str(merged), "--cuda", "--lazy-unpickle"],
                     {"CUDA_VISIBLE_DEVICES": str(gpus[0])}),
                    ([str(paths.TRAIN_PY), "scripts/finalize_merged_model.py", "--base", base, str(merged)], {})], log)
        (merged / "merge-source.json").write_text(json.dumps({"yaml": yaml_path.read_text(), "experts": {d: str(p) for d, p in experts.items()}}, indent=2))
        (merged / ".finalized").write_text("ok")

    sched.add({"id": f"merge:{name}", "gpus": {"min": 1, "max": 1}, "done": lambda: (merged / ".finalized").exists(), "run": run})
    return f"merge:{name}"


def _complete(run_dir):
    try:
        return json.loads((Path(run_dir) / "status.json").read_text()).get("state") == "complete"
    except (OSError, ValueError):
        return False


# ---------------------------------------------------------------- the pool (mopd.pool): submit, don't run
PRIORITY = {"train": 50, "merge": 30, "eval": 20, "grade": 0}


def _fresh_guard(run_dir):
    """A command refusing to retrain into an incomplete run directory (it would mix two runs)."""
    code = ("import json,sys,pathlib;d=pathlib.Path(sys.argv[1]);s=d/'status.json';"
            "sys.exit(0 if not d.exists() or (d/'resume'/'meta.json').exists() or (s.exists() and json.loads(s.read_text()).get('state')=='complete') "
            "else 'incomplete run directory '+str(d)+' without saved state: inspect or move aside before retraining')")
    return [[str(paths.TRAIN_PY), "-c", code, str(run_dir)], {}]


def pool_jobs(study_path, priority_offset=0):
    """Every job of the study as pool job specs (mopd.pool.store): ids "<study>/<job>", GPU shapes
    instead of ids, declarative done checks -- the daemon decides where and when."""
    from mopd.evaluation.eval_commands import eval_commands, grade_command

    study, reg, runs = plan(study_path)
    name = study["study"]
    out = paths.ROOT / "results/studies" / name
    (out / "configs").mkdir(parents=True, exist_ok=True)
    log = lambda run: str(out / "logs" / f"{run}.log")
    pr = lambda kind: PRIORITY[kind] + priority_offset
    specs = []

    def evals(r, model, deps, steps):
        for suite in r["evals"]:
            e = reg["evals"][suite]
            for step in steps:
                ck = None if step is None else f"checkpoint-{step:04d}"
                ev_out = r["run_dir"] / "evaluations" / (e["protocol_id"] if ck is None else f"{ck}/{e['protocol_id']}")
                jid = f"{name}/eval:{r['name']}:{suite}" + (f":{step}" if ck else "")
                cmds = eval_commands(data=resolve.abs_path(e["data"]), out=ev_out, protocol_id=e["protocol_id"],
                                     max_new_tokens=e["max_new_tokens"], device=["{gpus}"], grade=False,
                                     **model, **({"checkpoint": ck} if ck else {}))
                specs.append({"id": jid, "commands": [list(c) for c in cmds], "log": log(r["name"]),
                              "gpus": {"min": 1, "max": len(paths.TRAIN_GPUS)}, "priority": pr("eval"), "preemptible": True,
                              "deps": deps, "done": {"kind": "generated", "path": str(ev_out)}})
                specs.append({"id": f"{name}/grade|{jid.split('/', 1)[1]}", "commands": [list(grade_command(resolve.abs_path(e["data"]), ev_out))],
                              "log": log(r["name"]), "gpus": 0, "priority": pr("grade"), "deps": [jid],
                              "done": {"kind": "file", "path": str(ev_out / "summary.json")}})

    for r in runs:
        if "merge" in r:
            yaml_path = out / "configs" / f"{r['name']}.yaml"
            lines = [f"# study {name} / {r['name']}", "merge_method: linear", "models:"]
            for d in sorted(r["merge"]):
                lines += [f"  - model: {r['merge'][d]}", "    parameters:", "      weight: 1.0"]
            yaml_path.write_text("\n".join(lines + ["parameters:", "  normalize: true", "dtype: bfloat16"]) + "\n")
            base = resolve.abs_path(reg["models"][study["base"].get("student", "qwen3-4b-base")]["path"])
            merged = r["run_dir"]
            record = ("import json,sys,pathlib;m=pathlib.Path(sys.argv[1]);"
                      "(m/'merge-source.json').write_text(json.dumps({'yaml':pathlib.Path(sys.argv[2]).read_text(),'experts':json.loads(sys.argv[3])},indent=2));"
                      "(m/'.finalized').write_text('ok')")
            specs.append({"id": f"{name}/merge:{r['name']}", "log": log(r["name"]), "gpus": {"count": 1}, "priority": pr("merge"),
                          "done": {"kind": "file", "path": str(merged / ".finalized")},
                          "commands": [[[str(paths.MERGEKIT), str(yaml_path), str(merged), "--cuda", "--lazy-unpickle"], {}],
                                       [[str(paths.TRAIN_PY), "scripts/finalize_merged_model.py", "--base", base, str(merged)], {}],
                                       [[str(paths.TRAIN_PY), "-c", record, str(merged), str(yaml_path),
                                         json.dumps({d: str(p) for d, p in r["merge"].items()})], {}]]})
            evals(r, {"model": str(merged), "model_version_id": f"merge:{name}/{r['name']}"}, [f"{name}/merge:{r['name']}"], [None])
            continue
        deps = []
        cfg = r["config"]
        if not r["reused"]:
            cfg_path = out / "configs" / f"{r['name']}.json"
            cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
            mopd = cfg["training_method"] == "mopd"
            full = cfg["student_backend"] == "hf_full"
            n = len(paths.TRAIN_GPUS) if (full or mopd) else 1
            nproc = cfg["student_gpus"] if mopd else n
            script = "scripts/train_multidomain_opd.py" if mopd else "scripts/train_sft.py"
            # SFT saves its full state and exits when the pool asks (TrainRun.save_resume), so the pool may
            # preempt it and resume it later, on another GPU count if need be; MOPD cannot resume yet
            specs.append({"id": f"{name}/train:{r['name']}", "log": log(r["name"]), "gpus": {"count": n}, "priority": pr("train"),
                          **({} if mopd else {"preemptible": True, "graceful": True, "stop_grace": 900}),
                          "done": {"kind": "status_complete", "path": str(r["run_dir"])},
                          "commands": [_fresh_guard(r["run_dir"]),
                                       [[str(paths.TRAIN_PY), "-m", "torch.distributed.run", "--standalone",
                                         f"--nproc_per_node={nproc}", script, "--config", str(cfg_path)], {}]]})
            deps = [f"{name}/train:{r['name']}"]
        steps = r["fields"].get("checkpoints") or study.get("checkpoints") or [cfg["steps"]]
        evals(r, {"model_version": str(r["run_dir"] / "model-version.json")}, deps, steps)
    return specs
