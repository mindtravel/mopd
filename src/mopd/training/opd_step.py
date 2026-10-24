"""Teacher scoring and student backward of one on-policy distillation step, for any rank count.

Rank 0 scores trajectories with the domain teachers as they arrive (teacher_logps, per domain
micro-batch); then every rank backpropagates its share of the step (sharded_step) -- one rank does
all of it. Per-sample loss weights (domain weight / domain count) are fixed by the step plan, so the
summed gradient does not depend on arrival order or grouping. A sharded student (mopd.training.fsdp)
reduces gradients inside every backward call, so all ranks issue the same number of calls; a rank
with fewer runs a zero-weight one.
"""

import time

import torch
import torch.distributed as dist

from mopd.protocol import teacher_prefix_ids
from mopd.training.batching import is_eos_token, logps, pack
from mopd.training.dual_template import policy_loss
from mopd.training.fsdp import local
from mopd.training.microbatch import by_budget


class BlockContribution:
    """Per-domain gradient norm when each domain's backward calls run as one block (sharded_step with
    per_domain): one snapshot of the gradient at a time -- for a full-parameter student every extra
    copy is the size of the rank's whole parameter shard. With several ranks each holds a disjoint
    shard and the squared norms are summed."""

    def __init__(self, params, device):
        self.params, self.device, self.norms, self._before = params, device, {}, None

    def before(self):
        self._before = [
            local(p.grad).detach().clone() if p.grad is not None else torch.zeros_like(local(p)) for p in self.params
        ]

    def after(self, domain):
        sq = sum(
            (local(p.grad).detach() - b).float().square().sum()
            for p, b in zip(self.params, self._before)
            if p.grad is not None
        )
        total = torch.as_tensor(sq, device=self.device, dtype=torch.float64)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(total)
        self.norms[domain] = float(total) ** 0.5
        self._before = None

    def l2(self, domain):
        return self.norms.get(domain, 0.0)


class OPDStepScorer:
    """One training step's scoring/backward state: timings, per-sample entries, contributions."""

    def __init__(
        self, *, config, student, student_tokenizer, teachers, params, scales, eos_ids, sync, step, contribution=None
    ):
        self.c, self.student, self.stok, self.teachers = config, student, student_tokenizer, teachers
        self.scales, self.eos_ids, self.sync, self.step = scales, eos_ids, sync, step
        self.contrib = contribution  # BlockContribution, or None: no per-domain gradient norms
        self.entries = {d: [] for d in config["domains"]}
        self.records = []  # rollout-log lines, written by the trainer
        self.seq_lengths = []  # prompt + answer tokens of every trained trajectory (for MFU)
        self.weighted_loss = {d: 0.0 for d in config["domains"]}
        self.teacher_seconds = 0.0
        self.student_seconds = 0.0

    def teacher_logps(self, domain, chunk):
        spec = self.c["domains"][domain]
        tt, teacher = self.teachers[domain]
        tseqs, tstarts = [], []
        for r in chunk:
            prefix = teacher_prefix_ids(
                tt, r["row"], spec.get("teacher_prompt_format", "chat_template_v1"), r["prefix"]
            )
            tseqs.append(prefix + r["answer"])
            tstarts.append(len(prefix))
            assert tseqs[-1][tstarts[-1] :] == r["answer"]
        tid, ta, tm = pack(tseqs, tstarts, tt.pad_token_id, spec["device"])
        self.sync()
        mark = time.perf_counter()
        with torch.no_grad():
            tp = [v.to(self.c["student_device"]) for v in logps(teacher, tid, ta, tm)]
        self.sync()
        self.teacher_seconds += time.perf_counter() - mark
        return tp

    def student_backward(self, domain, chunk, tp, weight=1.0, domains=None):
        """weight=0 runs the forward/backward without contributing (keeps sharded ranks in step).
        domains: per-sample domain when one micro-batch mixes domains (sharded_step's single pass);
        each sample's loss then carries its own domain scale."""
        domains = domains or [domain] * len(chunk)
        seqs = [r["prefix"] + r["answer"] for r in chunk]
        starts = [len(r["prefix"]) for r in chunk]
        sid, sa, sm = pack(seqs, starts, self.stok.pad_token_id, self.c["student_device"])
        self.student.train()
        mark = time.perf_counter()
        sp = logps(self.student, sid, sa, sm)
        total = 0.0
        for sv, tv, r, d in zip(sp, tp, chunk, domains):
            loss, info = policy_loss(sv, tv.to(sv.device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            scaled = loss * self.scales[d] * weight
            total = total + scaled
            if weight:
                self.weighted_loss[d] += scaled.item()
                self.seq_lengths.append(len(r["prefix"]) + len(r["answer"]))
                info.update(
                    loss=loss.item(), tokens=len(r["answer"]), truncated=not is_eos_token(r["answer"][-1], self.eos_ids)
                )
                self.entries[d].append(info)
                self.records.append(
                    {
                        "step": self.step,
                        "student_version": self.step - 1,
                        "domain": d,
                        "teacher": self.c["domains"][d]["teacher"],
                        "id": r["row"]["id"],
                        "prompt": r["row"]["prompt"],
                        "response": self.stok.decode(r["answer"]),
                        "response_ids": r["answer"],
                        **info,
                    }
                )
        total.backward()
        self.sync()
        self.student_seconds += time.perf_counter() - mark

    def sharded_step(self, trajectories, teacher_lps, rank, world_size, micro, micro_tokens=None, per_domain=False):
        """Every rank: backpropagate this rank's share of the step, in lockstep.
        trajectories/teacher_lps: {domain: [...]}, identical on every rank.

        Every backward call of a sharded student all-gathers the model (twice under ZeRO-3) and
        reduce-scatters fp32 gradients, ~2 s on this machine, so the step is planned for few calls:
        one pass over all domains, micro-batched by padded-token budget (micro_batch_tokens; micro caps
        the count, mopd.training.microbatch), trajectories assigned longest-first to the rank with the
        fewest tokens. per_domain (config per_domain_grad_norm) instead runs each domain's calls as a
        block to report its gradient norm -- a diagnostic that triples the calls. All ranks run the
        same number of calls; a rank with fewer re-runs one at zero weight."""
        length = lambda it: len(it[1]["prefix"]) + len(it[1]["answer"])
        tagged = {d: [(d, t, lp) for t, lp in zip(trajectories[d], teacher_lps[d])] for d in self.c["domains"]}
        blocks = list(tagged.values()) if per_domain else [[it for d in tagged for it in tagged[d]]]
        for block in blocks:
            items = sorted(block, key=length, reverse=True)
            shares, load = [[] for _ in range(world_size)], [0] * world_size
            for it in items:  # identical on every rank: deterministic greedy (LPT) balance
                r = min(range(world_size), key=lambda k: (load[k], k))
                shares[r].append(it)
                load[r] += length(it)
            chunks = [list(by_budget(sh, length, micro, micro_tokens)) for sh in shares]
            calls = max(len(c) for c in chunks)
            if per_domain:
                self.contrib.before()
            for i in range(calls):
                chunk = chunks[rank][i] if i < len(chunks[rank]) else None
                if chunk:
                    self.student_backward(
                        chunk[0][0],
                        [t for _, t, _ in chunk],
                        [lp for _, _, lp in chunk],
                        domains=[d for d, _, _ in chunk],
                    )
                else:  # this rank ran out: same collectives, no contribution
                    d, t, lp = items[-1]
                    self.student_backward(d, [t], [lp], weight=0.0)
            if per_domain:
                self.contrib.after(block[0][0])

    def gather(self, world_size):
        """Sharded runs: merge every rank's entries, records and losses onto rank 0's scorer."""
        if world_size == 1:
            return
        parts = [None] * world_size
        dist.all_gather_object(
            parts, (self.entries, self.records, self.weighted_loss, self.student_seconds, self.seq_lengths)
        )
        self.entries = {d: [e for p in parts for e in p[0][d]] for d in self.entries}
        self.records = [r for p in parts for r in p[1]]
        self.weighted_loss = {d: sum(p[2][d] for p in parts) for d in self.weighted_loss}
        self.student_seconds = max(p[3] for p in parts)
        self.seq_lengths = [n for p in parts for n in p[4]]

    def domain_stats(self):
        stats = {}
        for domain, entries in self.entries.items():
            contribution = self.contrib.l2(domain) if self.contrib is not None else None
            if contribution is not None and not contribution > 0:
                raise ValueError("Zero domain gradient: " + domain)
            n = len(entries)
            stats[domain] = {
                "grad_contribution_l2": contribution,
                "count": n,
                "loss": sum(e["loss"] for e in entries) / n,
                "weighted_loss": self.weighted_loss[domain],
                "sampled_log_ratio": sum(e["sampled_log_ratio"] for e in entries) / n,
                "clipped_fraction": sum(e["clipped_fraction"] for e in entries) / n,
                "tokens": sum(e["tokens"] for e in entries),
                "truncated": sum(e["truncated"] for e in entries),
            }
        return stats
