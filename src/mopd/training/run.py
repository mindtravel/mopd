"""Scaffolding shared by the training entry points (scripts/train_sft.py,
scripts/train_multidomain_opd.py): everything around the training signal.

  process group and device (torchrun, any rank count); run directory, config/status/provenance
  files; the student (LoRA, full-parameter sharded with FSDP, QAT) with its optimizer and
  schedule; per-step clip + update; metrics/status lines; checkpoints with their manifest;
  the end-of-run weight-change check; failure/stop status. Only rank 0 writes files.

    run = TrainRun(config)
    with run:                                   # status failed/stopped + manifest on any error
        run.start(batch_plan=plan)              # mkdir, config.json, status.json, provenance
        tok, scorer = run.build_student()       # run.model, run.scorer, run.params, run.opt, run.sched
        for step, rows in enumerate(plan, 1):
            lr = run.begin_step()               # zero_grad; returns the step's lr
            ... forward/backward on run.scorer ...
            grad = run.optimize()               # clip 1.0, finiteness check, opt + sched step
            run.log(step, {...})                # metrics.jsonl line + status.json (rank 0)
            run.checkpoint(step)                # every save_every and at the end; barrier
        run.complete()                          # weight delta > 0, status complete + manifest
"""

import json
import time
import traceback
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.distributed as dist

from mopd.evaluation.versioning import write_manifest
from mopd.runtime.artifacts import dump
from mopd.runtime.provenance import record as record_provenance
from mopd.students.registry import build_student, is_sharded, save_student
from mopd.training import fsdp
from mopd.training.distributed import dist_info
from mopd.training.fsdp import local
from mopd.training.schedule import build_scheduler


class TrainRun:
    def __init__(self, config):
        self.rank, self.world_size, dist_device = dist_info()
        self.main = self.rank == 0
        self.sharded = is_sharded(config)
        if self.sharded and self.world_size < 4 and not config.get("fsdp", {}).get("offload"):
            raise ValueError("full fine-tuning (fp32 weights + AdamW sharded) needs >= 4 GPUs, or fsdp.offload")
        if "student_gpus" in config and self.world_size != config["student_gpus"]:
            raise ValueError(f"launch with --nproc_per_node={config['student_gpus']}")
        self.c = {**config, "student_device": dist_device} if dist_device else config
        self.device = self.c["student_device"]
        self.out = Path(self.c["output"])
        self.ckpt = None
        self._files = ExitStack()

    # ------------------------------------------------------------ lifecycle
    def start(self, batch_plan=None, resuming=False):
        """Create the run directory with config.json, status.json (and batch_plan.json), record
        provenance; a resume keeps the existing directory and records its own provenance file."""
        def setup():
            self.out.mkdir(parents=True, exist_ok=False)
            dump(self.out / "config.json", self.c)
            dump(self.out / "status.json", {"state": "loading"})
            if batch_plan is not None:
                dump(self.out / "batch_plan.json", batch_plan)
            write_manifest(self.c, self.out, {"state": "loading", "step": 0})
            record_provenance(self.c, self.out)

        if not resuming:
            self.on_main(setup)

    def on_main(self, fn):
        """Run fn on rank 0 only, then make every rank share its outcome: if it raised, all ranks
        raise. A rank-0-only failure followed by a barrier otherwise leaves the other ranks waiting out
        the 10-minute NCCL timeout (2026-10-24: a provenance error hung a 4-rank MOPD run)."""
        error = None
        if self.main:
            try:
                fn()
            except Exception:
                error = traceback.format_exc()
        if self.world_size > 1:
            box = [error]
            dist.broadcast_object_list(box, src=0)
            error = box[0]
        if error:
            raise RuntimeError("rank 0 failed:\n" + error)

    def build_student(self):
        """The student to train and score with: a (PEFT) model, or for a sharded backend the
        FSDP-wrapped TokenLogProbs root. Also the optimizer, schedule and the initial-weight copy."""
        torch.manual_seed(self.c["seed"])  # identical on every rank: same adapter init
        tok, self.model = build_student(self.c)
        self.model.train()
        shard_args = {k: v for k, v in self.c.get("fsdp", {}).items() if k != "optimizer_threads"}  # optimize() reads it
        self.scorer = fsdp.shard(self.model, **shard_args) if self.sharded else self.model
        self.params = [p for p in self.scorer.parameters() if p.requires_grad]
        self.initial = [local(p).detach().cpu().clone() for p in self.params] if (self.main or self.sharded) else None
        self.opt = torch.optim.AdamW(self.params, lr=self.c["lr"])
        self.sched = build_scheduler(self.opt, self.c)
        return tok, self.scorer

    def log_file(self, name, mode="w"):
        """A line-buffered file under the run directory on rank 0 (None elsewhere), closed at exit."""
        return self._files.enter_context((self.out / name).open(mode, buffering=1)) if self.main else None

    # ------------------------------------------------------------ per step
    def begin_step(self):
        self.opt.zero_grad(set_to_none=True)
        return self.sched.get_last_lr()[0]

    def optimize(self):
        if self.c.get("fsdp", {}).get("offload"):
            grad = self._clip_offloaded(1.0)
        else:
            grad = torch.nn.utils.clip_grad_norm_(self.params, 1.0)
            grad = grad.full_tensor() if hasattr(grad, "full_tensor") else grad
        if not torch.isfinite(grad) or grad <= 0:
            raise ValueError("Invalid student gradient")
        mark = time.perf_counter()
        self.opt.step()  # on the CPU under fsdp.offload (threads: cpu_threads)
        self.optimizer_seconds = time.perf_counter() - mark
        self.sched.step()
        return float(grad)

    def cpu_threads(self):
        """torch threads for this rank: under fsdp.offload the AdamW step and the gradient
        accumulation run on the CPU, so 24 (fsdp.optimizer_threads); otherwise 2, the trainers' long-
        standing setting (torchrun sets OMP_NUM_THREADS=1). Elementwise ops: the same numbers."""
        fs = self.c.get("fsdp", {})
        return fs.get("optimizer_threads", 24) if fs.get("offload") else 2

    def _clip_offloaded(self, max_norm, eps=1e-6):
        """clip_grad_norm_ for gradients sharded in host memory (fsdp.offload): NCCL reduces only
        device tensors, so the local squared sums go to the GPU for the all-reduce. Same norm and
        same scaling as clip_grad_norm_."""
        grads = [p.grad for p in self.params if p.grad is not None]
        local = sum((g.to_local() if hasattr(g, "to_local") else g).float().pow(2).sum() for g in grads)
        total = torch.as_tensor(local, dtype=torch.float64, device=torch.cuda.current_device())
        if dist.is_initialized():
            dist.all_reduce(total)
        norm = total.sqrt()
        coef = torch.clamp(max_norm / (norm + eps), max=1.0).item()
        if coef < 1.0:
            for g in grads:
                g.mul_(coef)
        return norm.float().cpu()

    def log(self, step, report, metrics=None):
        if not self.main:
            return
        line = json.dumps(report)
        (metrics or self.metrics).write(line + "\n")
        print(line, flush=True)
        dump(self.out / "status.json", {"state": "training", **report})

    def checkpoint(self, step):
        """Save at every save_every and at the last step; a barrier ends every step either way."""
        c = self.c
        if step % c["save_every"] == 0 or step == c["steps"]:
            self.ckpt = self.out / f"checkpoint-{step:04d}"
            if self.sharded or self.main:
                save_student(self.scorer if self.sharded else self.model, c, self.ckpt, step, self.main)
            if self.main:
                if not self.sharded:  # a sharded optimizer's rank-0 state is a fragment; sharded runs do not resume
                    torch.save(
                        {
                            "step": step,
                            "optimizer": self.opt.state_dict(),
                            "scheduler": self.sched.state_dict(),
                            "rng": torch.get_rng_state(),
                            "cuda_rng": torch.cuda.get_rng_state_all(),
                        },
                        self.ckpt / "training_state.pt",
                    )
                write_manifest(c, self.out, {"state": "training", "step": step})
        self.barrier()

    # ------------------------------------------------------------ end
    def weight_delta(self):
        """L2 norm of (final - initial) over every trainable parameter, summed over shards."""
        delta2 = sum(
            (local(p).detach().cpu() - v).float().square().sum().item() for p, v in zip(self.params, self.initial)
        )
        if self.sharded:
            total = torch.tensor(delta2, device=self.device, dtype=torch.float64)
            dist.all_reduce(total)
            delta2 = float(total)
        return delta2**0.5

    def complete(self, **extra):
        delta = self.weight_delta() if (self.main or self.sharded) else None
        if self.main:
            assert delta > 0
            dump(
                self.out / "status.json",
                {
                    "state": "complete",
                    "steps": self.c["steps"],
                    **extra,
                    "adapter_delta_l2": delta,
                    "final_adapter": str(self.ckpt),
                },
            )
            write_manifest(self.c, self.out, {"state": "complete", "step": self.c["steps"]})

    def barrier(self):
        if self.world_size > 1:
            dist.barrier()

    def __enter__(self):
        self.metrics = None
        return self

    def __exit__(self, kind, error, tb):
        try:
            if error is not None and self.main:
                state = "stopped" if isinstance(error, (SystemExit, KeyboardInterrupt)) else "failed"
                dump(
                    self.out / "status.json",
                    {"state": state, "traceback": "".join(traceback.format_exception(kind, error, tb))},
                )
                write_manifest(self.c, self.out, {"state": state})
        finally:
            self._files.close()
            if self.world_size > 1:
                dist.destroy_process_group()
        return False
