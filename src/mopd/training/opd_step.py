"""Score one micro-batch with its domain teacher and backpropagate the student loss.

Shared by streaming and whole-step rollouts: the trainer hands over (domain, micro-batch)
pairs in whatever order they become available; per-sample loss weights (domain weight /
domain count) are fixed by the step plan, so the accumulated gradient is order-independent.
"""
import json
import time

import torch

from mopd.protocol import teacher_prefix_ids
from mopd.training.batching import is_eos_token, logps, pack
from mopd.training.dual_template import policy_loss


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


class OPDStepScorer:
    """One training step's scoring/backward state: timings, per-sample entries, contributions."""

    def __init__(self, *, config, student, student_tokenizer, teachers, params, scales, eos_ids, sync, rollout_log, step):
        self.c, self.student, self.stok, self.teachers = config, student, student_tokenizer, teachers
        self.scales, self.eos_ids, self.sync, self.log, self.step = scales, eos_ids, sync, rollout_log, step
        self.contrib = GradContribution(params, config["domains"])
        self.entries = {d: [] for d in config["domains"]}
        self.weighted_loss = {d: 0.0 for d in config["domains"]}
        self.teacher_seconds = 0.0
        self.student_seconds = 0.0

    def process(self, domain, chunk):
        spec = self.c["domains"][domain]
        tt, teacher = self.teachers[domain]
        seqs = [r["prefix"] + r["answer"] for r in chunk]
        starts = [len(r["prefix"]) for r in chunk]
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
        del tid, ta, tm
        sid, sa, sm = pack(seqs, starts, self.stok.pad_token_id, self.c["student_device"])
        self.student.train(); mark = time.perf_counter()
        sp = logps(self.student, sid, sa, sm); losses = []
        for sv, tv, r in zip(sp, tp, chunk):
            loss, info = policy_loss(sv, tv)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            losses.append(loss)
            info.update(loss=loss.item(), tokens=len(r["answer"]), truncated=not is_eos_token(r["answer"][-1], self.eos_ids))
            self.entries[domain].append(info)
            self.log.write(json.dumps({"step": self.step, "student_version": self.step - 1, "domain": domain,
                "teacher": spec["teacher"], "id": r["row"]["id"], "prompt": r["row"]["prompt"],
                "response": self.stok.decode(r["answer"]), "response_ids": r["answer"], **info}, ensure_ascii=False) + "\n")
        domain_loss = torch.stack(losses).sum() * self.scales[domain]
        self.weighted_loss[domain] += domain_loss.item()
        self.contrib.before(); domain_loss.backward(); self.contrib.after(domain)
        self.sync(); self.student_seconds += time.perf_counter() - mark

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
