"""Synchronous multi-teacher OPD; shared on-policy student, fixed domain quotas. Any student backend
(LoRA, full-parameter sharded with FSDP, QAT): see run(). Scaffolding shared with SFT lives in
mopd.training.run.TrainRun."""

import argparse, hashlib, json, signal, sys, time
from pathlib import Path
import torch
import torch.distributed as dist
from mopd.backends.multi_teacher import load_teachers
from mopd.config import load
from mopd.data.mixture import plan_domains
from mopd.data.multidomain import domain_scale
from mopd.rollout.base import RolloutStream, open_stream
from mopd.rollout.hf_batch import generate
from mopd.rollout.policy import publish, cleanup as policy_cleanup
from mopd.rollout.processes import ProcessRollouts
from mopd.rollout.replicas import RolloutReplicas
from mopd.runtime.artifacts import dump, fingerprint
from mopd.runtime.provenance import record as record_provenance
from mopd.runtime.profile import ModelShape, opd_utilization
from mopd.training.batching import generation_eos_ids
from mopd.training.opd_step import OPDStepScorer, DomainBlockContribution, NO_CONTRIBUTION
from mopd.training.run import TrainRun
from mopd.training.streaming import DomainBatcher


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sync(c, main=True):
    """Teacher GPUs belong to rank 0 only."""
    for device in {c["student_device"], *((d["device"] for d in c["domains"].values()) if main else ())}:
        torch.cuda.synchronize(device)


def plan(c):
    return plan_domains(c)


# Runtime knobs a resume may change: they alter scheduling and batching only, not the objective,
# data, or optimizer (verified before use: greedy rollouts batch-invariant, gradients micro-batch-invariant).
RESUMABLE_OVERRIDES = {
    "generation_batch_size": int,
    "micro_batch_size": int,
    "rollout_devices": lambda v: v.split(","),
    "rollout_streaming": lambda v: {"true": True, "false": False}[v.lower()],
}


def prepare_resume(c, out, batches, resume):
    """Continue an interrupted LoRA run in place: cut logs back to the checkpoint, set aside newer
    rollout adapters, return the config to resume with and the step to resume from."""
    ck = Path(resume["checkpoint"])
    start = int(ck.name.rsplit("-", 1)[1])
    if ck.parent.resolve() != out.resolve():
        raise ValueError("checkpoint does not belong to this run")
    if json.loads((out / "batch_plan.json").read_text()) != json.loads(json.dumps(batches)):
        raise ValueError("re-planned batches differ from the run's batch_plan.json")
    for name in (
        "metrics.jsonl",
        "rollouts.jsonl",
    ):  # keep the interrupted run's full logs (never overwrite an earlier backup)
        f = out / name
        raw = f.read_text()
        backup = out / f"{name}.before-resume-{start:04d}"
        if not backup.exists():
            backup.write_text(raw)
        f.write_text("".join(l + "\n" for l in raw.splitlines() if l.strip() and json.loads(l)["step"] <= start))
    # rollout adapters are written once per policy version (version = step - 1); move the interrupted
    # run's versions >= start aside so the resumed steps can write them again
    runtime = out / "vllm_runtime"
    stale = runtime / f"stale-before-resume-{start:04d}"
    for d in sorted(runtime.glob("adapter-*")) if runtime.exists() else []:
        if int(d.name.split("-")[1]) >= start:
            stale.mkdir(exist_ok=True)
            d.rename(stale / d.name)
    c = {**c, **resume["overrides"], "student_adapter": str(ck)}
    record_provenance(c, out, filename=f"provenance-resume-{start:04d}.json")
    dump(
        out / f"resume-{start:04d}.json",
        {
            "checkpoint": str(ck),
            "checkpoint_sha256": sha256_file(ck / "adapter_model.safetensors"),
            "overrides": resume["overrides"],
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
    return c, start


def run(c, resume=None):
    """resume: {"checkpoint": Path, "overrides": {...}} continues an interrupted run in place (LoRA only).

    Launched through torchrun. One rank: the single-process trainer (rollouts streamed into teacher
    scoring and student backward). Several ranks: a sharded full-parameter student (student_backend
    hf_full, mopd.training.fsdp); rank 0 owns the rollout workers and teachers, scores the whole
    step, broadcasts it, and every rank backpropagates its share (opd_step)."""
    batches = plan(c)
    out = Path(c["output"])
    start = 0
    if resume:
        c, start = prepare_resume(c, out, batches, resume)
    run = TrainRun(c)
    c = run.c
    main, sharded, rank, world_size = run.main, run.sharded, run.rank, run.world_size
    if world_size > 1 and not sharded:
        raise ValueError("several ranks need a sharded (hf_full) student")
    if sharded and resume:
        raise ValueError("resume is LoRA-only for now")
    replicas = None
    with run:
        run.start(batch_plan=batches, resuming=bool(resume))
        torch.set_num_threads(2)
        stok, student = run.build_student()  # what logps() runs and the optimizer updates
        try:
            teachers = load_teachers(c, stok) if main else {}
            eos_ids = generation_eos_ids(run.model, stok)
            backend = c.get("rollout_backend", "replicas")
            factories = {"processes": ProcessRollouts, "replicas": RolloutReplicas}
            if backend == "vllm":
                from mopd.rollout.vllm_processes import VllmRollouts

                factories["vllm"] = VllmRollouts
            if backend not in factories and backend != "hf":
                raise ValueError("Unknown rollout_backend: " + str(backend))
            if main and backend != "hf" and c.get("rollout_devices"):
                replicas = factories[backend](c, student)
            params = run.params
            shape = ModelShape.of(student)
            hashes = {d: fingerprint(t) for d, (_, t) in teachers.items()}
            weights = {d: s["weight"] for d, s in c["domains"].items()}
            counts = {d: s["count"] for d, s in c["domains"].items()}
            scales = {d: domain_scale(d, weights, counts) for d in counts}
            if resume:
                state = torch.load(Path(resume["checkpoint"]) / "training_state.pt", weights_only=False)
                if state["step"] != start:
                    raise ValueError("training_state step mismatch")
                run.opt.load_state_dict(state["optimizer"])
                run.sched.load_state_dict(state["scheduler"])
                torch.set_rng_state(state["rng"])
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            mode = "a" if resume else "w"
            run.metrics = run.log_file("metrics.jsonl", mode)
            rollouts = run.log_file("rollouts.jsonl", mode)
            for step, rows in enumerate(batches, 1):
                if step <= start:
                    continue
                started = time.perf_counter()
                lr = run.begin_step()
                if main:
                    dump(out / "status.json", {"state": "training", "step": step, "phase": "generating"})
                mark = time.perf_counter()
                # vLLM workers run a published copy of the weights; publishing a sharded student is a collective
                policy = (
                    publish(student, c, out / "vllm_runtime", step - 1, stok, main) if backend == "vllm" else student
                )
                publish_seconds = time.perf_counter() - mark
                scorer = OPDStepScorer(
                    config=c,
                    student=student,
                    student_tokenizer=stok,
                    teachers=teachers,
                    params=params,
                    scales=scales,
                    eos_ids=eos_ids,
                    sync=lambda: sync(c, main),
                    step=step,
                    contribution=(
                        (
                            DomainBlockContribution(params, c["student_device"])
                            if c.get("per_domain_grad_norm")
                            else NO_CONTRIBUTION
                        )
                        if sharded
                        else None
                    ),
                )
                planned = {d: sum(r["domain"] == d for r in rows) for d in c["domains"]}
                if not sharded:
                    # rollouts -> per-domain micro-batches (arrival order) -> teacher score + student backward
                    if replicas:
                        stream = open_stream(
                            replicas, policy, rows, step - 1, streaming=bool(c.get("rollout_streaming"))
                        )
                    else:
                        trajectories, gen_seconds = generate(
                            student,
                            stok,
                            rows,
                            c["student_device"],
                            c["max_new_tokens"],
                            c["generation_batch_size"],
                            prompt_format=c["student_prompt_format"],
                        )
                        stream = RolloutStream(trajectories)
                        stream._started -= gen_seconds
                    for domain, chunk in DomainBatcher(c["micro_batch_size"], planned).consume(stream):
                        scorer.process(domain, chunk)
                    gen_seconds = stream.seconds
                    rollout_stats = stream.stats or {}
                else:
                    # rank 0: whole-step rollouts and teacher scores -> every rank: its share of the backward
                    payload = [None]
                    if main:
                        stream = open_stream(replicas, policy, rows, step - 1, streaming=False)
                        by_domain = {d: [] for d in c["domains"]}
                        for t in stream:
                            by_domain[t["row"]["domain"]].append(t)
                        lps = {d: [] for d in c["domains"]}
                        for d, items in by_domain.items():
                            for i in range(0, len(items), c["micro_batch_size"]):
                                lps[d] += [
                                    v.cpu() for v in scorer.teacher_logps(d, items[i : i + c["micro_batch_size"]])
                                ]
                        payload = [(by_domain, lps, stream.seconds, stream.stats or {}, scorer.teacher_seconds)]
                    dist.broadcast_object_list(payload, src=0)
                    by_domain, lps, gen_seconds, rollout_stats, scorer.teacher_seconds = payload[0]
                    scorer.sharded_step(
                        by_domain,
                        lps,
                        rank,
                        world_size,
                        c["micro_batch_size"],
                        c.get("micro_batch_tokens"),
                        per_domain=c.get("per_domain_grad_norm", False),
                    )
                    scorer.gather(world_size)
                stats = scorer.domain_stats() if main else None
                grad = run.optimize()
                sync(c, main)
                if main:
                    for record in scorer.records:
                        rollouts.write(json.dumps(record, ensure_ascii=False) + "\n")
                    seconds = time.perf_counter() - started
                    util = opd_utilization(
                        shape,
                        scorer.seq_lengths,
                        student_seconds=scorer.student_seconds,
                        step_seconds=seconds,
                        generation_seconds=gen_seconds,
                        student_gpus=world_size,
                        rollout_gpus=len(c.get("rollout_devices", [])),
                        generated_tokens=sum(s["tokens"] for s in stats.values()),
                        recompute=c.get("gradient_checkpointing", True),
                    )
                    report = {
                        "step": step,
                        "lr": lr,
                        "domains": stats,
                        "loss": sum(s["weighted_loss"] for s in stats.values()),
                        "grad_norm": grad,
                        "rollout": rollout_stats,
                        "publish_seconds": publish_seconds,
                        "generation_seconds": gen_seconds,
                        "teacher_seconds": scorer.teacher_seconds,
                        "student_seconds": scorer.student_seconds,
                        "seconds": seconds,
                        "generated_tokens_per_second": sum(s["tokens"] for s in stats.values()) / gen_seconds,
                        "student_peak_gib": torch.cuda.max_memory_allocated(c["student_device"]) / 2**30,
                        "world_size": world_size,
                        "utilization": util,
                    }
                    run.log(step, report)
                run.checkpoint(step)
            if hasattr(replicas, "close"):
                replicas.close()
            extra = {}
            if main:
                unchanged = {
                    d: hashes[d] == fingerprint(t) and all(p.grad is None for p in t.parameters())
                    for d, (_, t) in teachers.items()
                }
                assert all(unchanged.values())
                extra["teachers_unchanged"] = unchanged
            run.complete(**extra)
        finally:
            if hasattr(replicas, "close"):
                replicas.close()
            if main:
                policy_cleanup(c)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume-from", help="checkpoint directory of this run to continue from")
    p.add_argument(
        "--override", action="append", default=[], help=f"KEY=VALUE, KEY in {sorted(RESUMABLE_OVERRIDES)} (resume only)"
    )
    a = p.parse_args()
    if a.override and not a.resume_from:
        p.error("--override is only allowed with --resume-from")
    overrides = {}
    for item in a.override:
        k, _, v = item.partition("=")
        if k not in RESUMABLE_OVERRIDES:
            p.error(f"{k} is not a resumable runtime knob")
        overrides[k] = RESUMABLE_OVERRIDES[k](v)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit("SIGTERM"))
    if a.resume_from:
        c = load(Path(a.resume_from).parent / "config.json")  # the run's own recorded config, not the spec copy
        run(c, {"checkpoint": a.resume_from, "overrides": overrides})
    else:
        run(load(a.config))
