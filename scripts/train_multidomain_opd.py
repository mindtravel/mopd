"""Synchronous multi-teacher OPD; shared on-policy student, fixed domain quotas. Any student backend
(LoRA, full-parameter sharded with FSDP, QAT): see run(). Scaffolding shared with SFT lives in
mopd.training.run.TrainRun."""

import argparse, hashlib, json, signal, sys, time, traceback
from pathlib import Path
import torch
import torch.distributed as dist
from mopd.backends.multi_teacher import load_teachers
from mopd.config import load
from mopd.data.mixture import plan_domains
from mopd.data.multidomain import domain_scale
from mopd.rollout.base import Prefetched, RolloutStream, open_stream
from mopd.rollout.hf_batch import generate
from mopd.rollout.policy import Publisher
from mopd.rollout.processes import ProcessRollouts
from mopd.rollout.replicas import RolloutReplicas
from mopd.runtime.artifacts import dump, fingerprint
from mopd.runtime.provenance import record as record_provenance
from mopd.profiling.model import ModelShape, opd_utilization
from mopd.profiling.training import torch_profile
from mopd.training.batching import generation_eos_ids
from mopd.training.opd_step import OPDStepScorer, BlockContribution
from mopd.training.run import TrainRun
from mopd.training.streaming import DomainBatcher


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sync(c, main=True):
    """Teacher GPUs belong to rank 0 only."""
    for device in {c["student_device"], *((d["device"] for d in c["domains"].values()) if main else ())}:
        torch.cuda.synchronize(device)


def load_replay(path):
    """{step: {row id: answer token ids}} from a run's rollouts.jsonl."""
    out = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        out.setdefault(r["step"], {})[r["id"]] = r["response_ids"]
    return out


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
    # rollout_async: step k trains on trajectories from policy version k-2 (version 0 for steps 1-2),
    # generated on the rollout GPUs while step k-1's student update runs; off-policy by one step
    asynchronous = bool(c.get("rollout_async"))
    if asynchronous and (resume or c.get("rollout_backend") != "vllm" or not c.get("rollout_devices") or not c.get("rollout_streaming")):
        raise ValueError("rollout_async needs streaming vLLM rollout GPUs and no resume")
    replicas = publisher = None  # the finally block below closes whichever exists
    with run:
        run.start(batch_plan=batches, resuming=bool(resume))
        torch.set_num_threads(run.cpu_threads())
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
            if main and backend != "hf" and c.get("rollout_devices") and not c.get("rollout_replay"):
                replicas = factories[backend](c, student)
            publisher = Publisher(c, out / "vllm_runtime", stok, main) if backend == "vllm" else None

            replay = load_replay(c["rollout_replay"]) if c.get("rollout_replay") else None

            def rollout_stream(policy, rows, step):
                """One step's trajectories as a stream (vLLM/replica workers, or the student itself).
                rollout_replay: another run's recorded answers for these rows instead (benchmarks:
                two configurations train on identical samples, so their timings compare)."""
                if replay is not None:
                    from mopd.data.domains import student_prompt

                    answers = replay.get(step)
                    if answers is None or any(r["id"] not in answers for r in rows):
                        raise ValueError(f"rollout_replay has no answers for every row of step {step}")
                    return RolloutStream([
                        {"index": i, "row": r, "answer": answers[r["id"]], "student_version": step - 1, "first_logprobs": None,
                         "prefix": stok.encode(student_prompt(stok, r, c["student_prompt_format"]))}
                        for i, r in enumerate(rows)])
                if replicas:
                    return open_stream(replicas, policy, rows, step - 1, streaming=bool(c.get("rollout_streaming")))
                trajectories, seconds = generate(
                    student,
                    stok,
                    rows,
                    c["student_device"],
                    c["max_new_tokens"],
                    c["generation_batch_size"],
                    prompt_format=c["student_prompt_format"],
                )
                stream = RolloutStream(trajectories)
                stream._started -= seconds
                return stream

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
            pending = None  # async: (stream, policy version) of the step about to train, already decoding
            if asynchronous:
                policy = publisher.publish(student, 0)
                pending = (Prefetched(rollout_stream(policy, batches[0], 1)) if main else None, 0)
            for step, rows in enumerate(batches, 1):
                if step <= start:
                    continue
                started = time.perf_counter()
                lr = run.begin_step()
                if main:
                    dump(out / "status.json", {"state": "training", "step": step, "phase": "generating"})
                mark = time.perf_counter()
                # vLLM workers run a published copy of the weights; publishing a sharded student is a collective
                if asynchronous:
                    publish_seconds = 0.0  # this step's stream was opened last step; publishing comes after it drains
                    policy_version = pending[1]
                else:
                    policy = publisher.publish(student, step - 1) if publisher else student
                    publish_seconds = time.perf_counter() - mark
                    policy_version = step - 1
                per_domain = c.get("per_domain_grad_norm", world_size == 1)  # free without collectives
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
                    contribution=BlockContribution(params, c["student_device"]) if per_domain else None,
                )
                # rank 0: rollouts, scored by the teachers as they arrive (streamed per domain micro-batch)
                # -> every rank: its share of the student backward (one rank: all of it)
                payload = [None]
                if main:
                    try:
                        stream = pending[0] if asynchronous else rollout_stream(policy, rows, step)
                        by_domain = {d: [] for d in c["domains"]}
                        lps = {d: [] for d in c["domains"]}
                        planned = {d: sum(r["domain"] == d for r in rows) for d in c["domains"]}
                        for domain, chunk in DomainBatcher(c["micro_batch_size"], planned).consume(stream):
                            by_domain[domain] += chunk
                            lps[domain] += [v.cpu() for v in scorer.teacher_logps(domain, chunk)]
                        payload = [(by_domain, lps, stream.seconds, stream.stats or {}, scorer.teacher_seconds)]
                        rollout_wait = time.perf_counter() - mark  # generation the student phase did not hide
                    except Exception:  # other ranks wait in the broadcast: send them the failure, not a hang
                        payload = [{"error": traceback.format_exc()}]
                if world_size > 1:
                    dist.broadcast_object_list(payload, src=0)
                if isinstance(payload[0], dict):
                    raise RuntimeError("rank 0 rollout/scoring failed:\n" + payload[0]["error"])
                by_domain, lps, gen_seconds, rollout_stats, scorer.teacher_seconds = payload[0]
                if asynchronous and step < len(batches):
                    # every rollout of the previous version has returned: rewrite the policy and start the
                    # next step decoding before this step's student update
                    mark = time.perf_counter()
                    version = max(step - 1, 0)
                    if version != pending[1]:
                        policy = publisher.publish(student, version)
                    publish_seconds = time.perf_counter() - mark
                    pending = (Prefetched(rollout_stream(policy, batches[step], version + 1)) if main else None, version)
                # profile_step: torch.profiler kernel breakdown (comm / matmul / attention / idle) of that
                # step's student phase on rank 0 (profiling.training.torch_profile)
                with torch_profile(main and c.get("profile_step") == step, out, step) as trace:
                    scorer.sharded_step(
                        by_domain,
                        lps,
                        rank,
                        world_size,
                        c["micro_batch_size"],
                        c.get("micro_batch_tokens"),
                        per_domain=per_domain,
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
                        seconds={"step": seconds, "student": scorer.student_seconds, "generation": gen_seconds},
                        gpus={"student": world_size, "rollout": len(c.get("rollout_devices", []))},
                        # vLLM engines count their own steps and tokens; other backends only report tokens
                        rollout=rollout_stats.get("counters") or {"generated": sum(s["tokens"] for s in stats.values())},
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
                        "policy_version": policy_version,
                        "rollout_wait_seconds": rollout_wait,
                        "generation_seconds": gen_seconds,
                        "teacher_seconds": scorer.teacher_seconds,
                        "student_seconds": scorer.student_seconds,
                        "optimizer_seconds": run.optimizer_seconds,
                        "seconds": seconds,
                        "generated_tokens_per_second": sum(s["tokens"] for s in stats.values()) / gen_seconds,
                        "student_peak_gib": torch.cuda.max_memory_allocated(c["student_device"]) / 2**30,
                        "world_size": world_size,
                        "utilization": util,
                        **({"kernels": trace} if trace else {}),
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
            if publisher:
                publisher.close()


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
