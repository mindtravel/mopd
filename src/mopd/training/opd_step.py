"""Score one micro-batch with its domain teacher and backpropagate the student loss.

Shared by streaming and whole-step rollouts: the trainer hands over (domain, micro-batch)
pairs in whatever order they become available; per-sample loss weights (domain weight /
domain count) are fixed by the step plan, so the accumulated gradient is order-independent.

A sharded student (several ranks, mopd.training.fsdp) splits the two halves: rank 0 scores every
trajectory with the teachers (teacher_logps), then every rank backpropagates its share of each
domain (sharded_step). FSDP reduces gradients inside each backward call, so ranks must issue the
same number of backward calls in the same domain order; a rank with fewer micro-batches in a domain
runs a zero-weight one instead.
"""
import time

import torch
import torch.distributed as dist

from mopd.protocol import teacher_prefix_ids
from mopd.training.batching import is_eos_token, logps, pack
from mopd.training.dual_template import policy_loss
from mopd.training.fsdp import local


class GradContribution:
    """Per-domain gradient contribution when domains' micro-batches interleave.

    Snapshot gradients before each backward and add the difference to that domain's
    accumulator; the norm equals the old "grad after domain minus grad before domain"."""

    def __init__(self, params, domains):
        self.params = params
        self.acc = {d: None for d in domains}
        self._before = None

    def before(self):
        self._before = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in self.params]

    def after(self, domain):
        delta = [p.grad.detach() - b for p, b in zip(self.params, self._before)]
        acc = self.acc[domain]
        self.acc[domain] = delta if acc is None else [a + d for a, d in zip(acc, delta)]
        self._before = None

    def l2(self, domain):
        acc = self.acc[domain]
        return 0.0 if acc is None else sum(a.float().square().sum().item() for a in acc) ** 0.5


class DomainBlockContribution:
    """Per-domain contribution when each domain's backward calls are contiguous (sharded_step):
    one snapshot at a time instead of an accumulator per domain -- for a full-parameter student,
    each extra copy of the gradient is the size of the rank's whole parameter shard. Norms are
    summed over ranks, each holding a disjoint shard."""

    def __init__(self, params, device):
        self.params, self.device, self.norms, self._before = params, device, {}, None

    def before(self):
        self._before = [local(p.grad).detach().clone() if p.grad is not None else torch.zeros_like(local(p)) for p in self.params]

    def after(self, domain):
        sq = sum((local(p.grad).detach() - b).float().square().sum() for p, b in zip(self.params, self._before) if p.grad is not None)
        total = torch.as_tensor(sq, device=self.device, dtype=torch.float64)
        dist.all_reduce(total)
        self.norms[domain] = float(total) ** 0.5
        self._before = None

    def l2(self, domain):
        return self.norms.get(domain, 0.0)


class OPDStepScorer:
    """One training step's scoring/backward state: timings, per-sample entries, contributions."""

    def __init__(self, *, config, student, student_tokenizer, teachers, params, scales, eos_ids, sync, step,
                 contribution=None):
        self.c, self.student, self.stok, self.teachers = config, student, student_tokenizer, teachers
        self.scales, self.eos_ids, self.sync, self.step = scales, eos_ids, sync, step
        self.contrib = contribution or GradContribution(params, config["domains"])
        self.entries = {d: [] for d in config["domains"]}
        self.records = []  # rollout-log lines, written by the trainer
        self.weighted_loss = {d: 0.0 for d in config["domains"]}
        self.teacher_seconds = 0.0
        self.student_seconds = 0.0

    def process(self, domain, chunk):
        self.student_backward(domain, chunk, self.teacher_logps(domain, chunk))

    def teacher_logps(self, domain, chunk):
        spec = self.c["domains"][domain]
        tt, teacher = self.teachers[domain]
        tseqs, tstarts = [], []
        for r in chunk:
            prefix = teacher_prefix_ids(tt, r["row"], spec.get("teacher_prompt_format", "chat_template_v1"), r["prefix"])
            tseqs.append(prefix + r["answer"]); tstarts.append(len(prefix))
            assert tseqs[-1][tstarts[-1]:] == r["answer"]
        tid, ta, tm = pack(tseqs, tstarts, tt.pad_token_id, spec["device"])
        self.sync(); mark = time.perf_counter()
        with torch.no_grad():
            tp = [v.to(self.c["student_device"]) for v in logps(teacher, tid, ta, tm)]
        self.sync(); self.teacher_seconds += time.perf_counter() - mark
        return tp

    def student_backward(self, domain, chunk, tp, weight=1.0):
        """weight=0 runs the forward/backward without contributing (keeps sharded ranks in step)."""
        spec = self.c["domains"][domain]
        seqs = [r["prefix"] + r["answer"] for r in chunk]
        starts = [len(r["prefix"]) for r in chunk]
        sid, sa, sm = pack(seqs, starts, self.stok.pad_token_id, self.c["student_device"])
        self.student.train(); mark = time.perf_counter()
        sp = logps(self.student, sid, sa, sm); losses = []
        for sv, tv, r in zip(sp, tp, chunk):
            loss, info = policy_loss(sv, tv.to(sv.device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            losses.append(loss)
            if weight:
                info.update(loss=loss.item(), tokens=len(r["answer"]), truncated=not is_eos_token(r["answer"][-1], self.eos_ids))
                self.entries[domain].append(info)
                self.records.append({"step": self.step, "student_version": self.step - 1, "domain": domain,
                    "teacher": spec["teacher"], "id": r["row"]["id"], "prompt": r["row"]["prompt"],
                    "response": self.stok.decode(r["answer"]), "response_ids": r["answer"], **info})
        domain_loss = torch.stack(losses).sum() * self.scales[domain] * weight
        self.weighted_loss[domain] += domain_loss.item()
        if isinstance(self.contrib, GradContribution):
            self.contrib.before(); domain_loss.backward(); self.contrib.after(domain)
        else:
            domain_loss.backward()
        self.sync(); self.student_seconds += time.perf_counter() - mark

    def sharded_step(self, trajectories, teacher_lps, rank, world_size, micro):
        """Every rank: backpropagate this rank's share of each domain, domain by domain, in lockstep.
        trajectories/teacher_lps: {domain: [...]}, identical on every rank."""
        for domain in self.c["domains"]:
            items = list(zip(trajectories[domain], teacher_lps[domain]))
            mine = items[rank::world_size]
            calls = -(-max(len(items[r::world_size]) for r in range(world_size)) // micro)
            self.contrib.before()
            for i in range(calls):
                chunk = mine[i * micro:(i + 1) * micro]
                if chunk:
                    self.student_backward(domain, [t for t, _ in chunk], [lp for _, lp in chunk])
                else:  # this rank ran out: same collectives, no contribution
                    self.student_backward(domain, [items[0][0]], [items[0][1]], weight=0.0)
            self.contrib.after(domain)

    def gather(self, world_size):
        """Sharded runs: merge every rank's entries, records and losses onto rank 0's scorer."""
        if world_size == 1:
            return
        parts = [None] * world_size
        dist.all_gather_object(parts, (self.entries, self.records, self.weighted_loss, self.student_seconds))
        self.entries = {d: [e for p in parts for e in p[0][d]] for d in self.entries}
        self.records = [r for p in parts for r in p[1]]
        self.weighted_loss = {d: sum(p[2][d] for p in parts) for d in self.weighted_loss}
        self.student_seconds = max(p[3] for p in parts)

    def domain_stats(self):
        stats = {}
        for domain, entries in self.entries.items():
            contribution = self.contrib.l2(domain)
            if not contribution > 0:
                raise ValueError("Zero domain gradient: " + domain)
            n = len(entries)
            stats[domain] = {"grad_contribution_l2": contribution, "count": n, "loss": sum(e["loss"] for e in entries) / n,
                "weighted_loss": self.weighted_loss[domain],
                "sampled_log_ratio": sum(e["sampled_log_ratio"] for e in entries) / n,
                "clipped_fraction": sum(e["clipped_fraction"] for e in entries) / n,
                "tokens": sum(e["tokens"] for e in entries), "truncated": sum(e["truncated"] for e in entries)}
        return stats
