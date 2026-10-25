"""Plain supervised fine-tuning on real (prompt, reference answer) pairs.

Everything except the training signal is shared with the MOPD entry
(scripts/train_multidomain_opd.py) through mopd.training.run.TrainRun, so a comparison isolates
the method:
  - batches: mopd.data.mixture.resolve_domain + plan_domains (fixed per-step domain
    and source quotas, no replacement, holdout overlap rejected before training);
  - loss weighting: per-sample token mean, then per-domain mean, then domain weights,
    the reduction MOPD applies via mopd.data.multidomain.domain_scale;
  - optional "token_budget": {"per_step": N} sizes each step by trained tokens instead of
    sample count (mopd.data.token_budget), same pools and shuffle order;
  - optimizer and schedule: AdamW + mopd.training.schedule, grad clip 1.0;
  - student, prompt protocol, packed forward, checkpoint and manifest.
The only difference is that there is no rollout and no teacher: each sample's loss
is token-level negative log-likelihood of its fixed reference answer.

Does not go through mopd.config.load: that schema requires a teacher and a rollout
backend per domain, which do not exist for SFT.

Multi-GPU: `torchrun --standalone --nproc_per_node=N scripts/train_sft.py --config ...`.
  LoRA students data-parallelize: each rank runs the single-GPU forward/backward on its own shard
  of the step's batch, then every trainable parameter's .grad is summed (not averaged) across
  ranks. Every rank scales its per-sample loss by the *global* per-step domain weight (computed
  from the full, unsharded batch), so summing the disjoint per-rank shards reproduces exactly the
  gradient a single GPU would compute over the whole step. Not DistributedDataParallel: logps()
  bypasses the student's top-level forward, so DDP's hooks would never fire.
  Full-parameter students (student_backend hf_full) are sharded with FSDP instead
  (mopd.training.fsdp), which reduce-scatters the same SUM inside backward.
Reported per-step metrics are summed across ranks so they describe the whole step.
"""

import argparse, sys, json, time
from pathlib import Path
import torch
import torch.distributed as dist
from mopd.data.mixture import resolve_domain, plan_domains
from mopd.data.token_budget import plan_by_tokens
from mopd.protocol import student_prompt
from mopd.runtime.artifacts import dump
from mopd.profiling.model import ModelShape
from mopd.profiling.training import StepProfile, torch_profile
from mopd.training import fsdp
from mopd.training.batching import pack, logps
from mopd.training.microbatch import by_budget
from mopd.training.run import TrainRun
from mopd.training.schedule import describe


def encode(tok, row, prompt_format, eos, max_len):
    prefix = tok(student_prompt(tok, row, prompt_format), add_special_tokens=False).input_ids
    response = tok(row["response"], add_special_tokens=False).input_ids + [eos]
    if max_len and len(prefix) + len(response) > max_len:
        return prefix, response[: max_len - len(prefix)], True
    return prefix, response, False


def sync_stats(stats, device):
    """Sum each domain's local sample/loss/token/truncated counters across ranks: every rank only
    processed a shard of the step, so the reported metrics must cover the whole step, not one shard."""
    domains = sorted(stats)
    packed = torch.tensor(
        [[stats[d]["samples"], stats[d]["loss_sum"], stats[d]["tokens"], stats[d]["truncated"]] for d in domains],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    return {
        d: {
            "samples": int(packed[i, 0]),
            "loss_sum": float(packed[i, 1]),
            "tokens": int(packed[i, 2]),
            "truncated": int(packed[i, 3]),
        }
        for i, d in enumerate(domains)
    }


PREEMPTED = 75  # exit code the pool reads as "stopped on request; requeue" (mopd.pool.daemon)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    c = json.loads(Path(a.config).read_text())
    for name, spec in c["domains"].items():
        resolve_domain(name, spec)
    describe(c)  # pure validation, no side effect; run on every rank
    run = TrainRun(c)
    c = run.c
    with run:
        run.start()
        tok, scorer = run.build_student()
        device = run.device
        pad = tok.pad_token_id
        eos = tok.eos_token_id
        max_len = c.get("max_seq_len")
        micro = c["micro_batch_size"]
        fmt = c["student_prompt_format"]
        if c.get("token_budget"):
            plan = plan_by_tokens(c, lambda row: len(encode(tok, row, fmt, eos, max_len)[1]))
        else:
            plan = plan_domains(c)  # identical on every rank: deterministic from config["seed"] + data files
        if run.world_size > 1 and any(len(rows) < run.world_size for rows in plan):
            raise ValueError(
                f"samples_per_step is smaller than world_size={run.world_size}: some ranks would get no rows"
            )
        if run.main and not run.start_step:
            dump(run.out / "batch_plan.json", plan)
        # resumed (the pool preempted this run, mopd.pool): weights, optimizer, scheduler and RNG from
        # resume/, the same deterministic plan, logs cut back to the saved step
        start = run.restore()
        run.truncate_logs("metrics.jsonl")
        weights = {d: s["weight"] for d, s in c["domains"].items()}
        run.metrics = run.log_file("metrics.jsonl", "a" if start else "w")
        shape = ModelShape.of(scorer)
        recompute = c.get("gradient_checkpointing", True)
        for step, rows in enumerate(plan, 1):
            if step <= start:
                continue
            # profile_step: dump a torch.profiler kernel breakdown (comm / matmul / attention / idle) of
            # that one step on rank 0 -- the per-step StepProfile cannot see inside FSDP's hooks.
            with torch_profile(run.main and c.get("profile_step") == step, run.out, step) as trace:
                prof = StepProfile(shape, run.world_size, recompute=recompute, device=device)
                started = time.perf_counter()
                lr = run.begin_step()
                stats = {d: {"samples": 0, "loss_sum": 0.0, "tokens": 0, "truncated": 0} for d in weights}
                # Per-sample weight = domain weight / that domain's samples this step: per-sample token
                # mean, then domain mean, then domain weights (MOPD's domain_scale when counts are fixed).
                # Computed from the full step (all ranks' rows), not the local shard below, so summed
                # per-rank gradients reproduce the single-GPU gradient exactly (see module docstring).
                counts = {d: sum(r["domain"] == d for r in rows) for d in weights}
                scales = {d: weights[d] / counts[d] for d in weights}
                local_rows = rows[run.rank :: run.world_size]  # disjoint, deterministic shard of this step's batch
                # Encode the shard once and micro-batch it longest-first, by count (micro_batch_size) and
                # optionally by padded-token budget (micro_batch_tokens, mopd.training.microbatch): in arrival
                # order half the step's tokens were padding (profile: 0.50). The gradient does not depend on
                # the grouping (fixed per-sample scales, summed), only the work and the peak memory.
                encoded = [encode(tok, row, fmt, eos, max_len) + (row,) for row in local_rows]
                chunks = list(by_budget(encoded, lambda e: len(e[0]) + len(e[1]), micro, c.get("micro_batch_tokens")))
                real = len(chunks)
                if run.sharded and run.world_size > 1:
                    # FSDP issues per-layer collectives inside every backward, so every rank must run the
                    # same number of micro-batches: ranks with fewer re-run their shortest one at zero weight.
                    most = torch.tensor(real, device=device)
                    dist.all_reduce(most, op=dist.ReduceOp.MAX)
                    chunks += [chunks[-1][-1:]] * (int(most) - real)
                for n, chunk in enumerate(chunks):
                    dummy = n >= real
                    fsdp.before_micro_batch(scorer, last=n == len(chunks) - 1)
                    items = [e[3] for e in chunk]
                    seqs = []
                    starts = []
                    truncated = []
                    for prefix, response, cut, row in chunk:
                        if not response:
                            raise ValueError(f"prompt alone exceeds max_seq_len: {row['id']}")
                        seqs.append(prefix + response)
                        starts.append(len(prefix))
                        truncated.append(cut)
                    ids, attention, response_mask = pack(seqs, starts, pad, device)
                    if not dummy:
                        prof.add_batch([len(q) for q in seqs], ids.shape[1])
                    with prof.phase("forward_backward"):
                        per_sample = [-v.mean() for v in logps(scorer, ids, attention, response_mask)]
                        loss = sum(scales[r["domain"]] * l for r, l in zip(items, per_sample)) * (0.0 if dummy else 1.0)
                        loss.backward()
                    if dummy:
                        continue
                    for r, l, s, start, cut in zip(items, per_sample, seqs, starts, truncated):
                        d = stats[r["domain"]]
                        d["samples"] += 1
                        d["loss_sum"] += float(l.detach())
                        d["tokens"] += len(s) - start
                        d["truncated"] += int(cut)
                peak = torch.tensor(torch.cuda.max_memory_allocated(device) / 2**30, device=device)
                # whole-step token accounting for MFU: every rank's real tokens, padded widths and squared lengths
                lengths = torch.tensor(
                    [sum(prof.lengths), prof.padded, sum(q * q for q in prof.lengths)],
                    device=device,
                    dtype=torch.float64,
                )
                with prof.phase("gradient_sync"):
                    if run.world_size > 1:
                        dist.all_reduce(peak, op=dist.ReduceOp.MAX)  # the rank closest to OOM, not rank 0
                        dist.all_reduce(lengths)
                        if not run.sharded:  # FSDP already reduce-scattered sharded gradients (as a SUM, see fsdp.py)
                            for p in run.params:
                                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        stats = sync_stats(stats, device)
                with prof.phase("optimizer"):
                    grad = run.optimize()
                if run.main:
                    domains = {
                        d: {
                            "loss": s["loss_sum"] / s["samples"],
                            "samples": s["samples"],
                            "tokens": s["tokens"],
                            "truncated": s["truncated"],
                        }
                        for d, s in stats.items()
                    }
                    report = {
                        "step": step,
                        "lr": lr,
                        "loss": sum(weights[d] * v["loss"] for d, v in domains.items()),
                        "grad_norm": grad,
                        "domains": domains,
                        "tokens": sum(v["tokens"] for v in domains.values()),
                        "truncated": sum(v["truncated"] for v in domains.values()),
                        "seconds": time.perf_counter() - started,
                        "world_size": run.world_size,
                        "max_memory_gib": round(float(peak), 2),
                        "profile": prof.finish(
                            total_tokens=int(lengths[0]), padded=int(lengths[1]), sum_sq=float(lengths[2])
                        ),
                        **({"kernels": trace} if trace else {}),
                    }
                    run.log(step, report)
            run.checkpoint(step)
            if step < len(plan) and run.stop_requested():  # preempted: save everything, exit as "preempted"
                run.save_resume(step)
                sys.exit(PREEMPTED)
        run.complete()


if __name__ == "__main__":
    main()
