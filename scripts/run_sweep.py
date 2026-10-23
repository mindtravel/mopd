"""Data-ablation sweep: one SFT expert per recipe, scored on a dev benchmark, collected in a leaderboard.

  configs/sweeps/<name>.json
  {"name", "description",
   "controls": {"student", "seed", "steps", "samples_per_step", "lr", "lr_schedule",
                "save_every",                      # max_seq_len and micro_batch_size come from each recipe
                "student_backend",                 # optional: "hf_lora" (default) or "hf_full"
                "fsdp", "gradient_checkpointing"}, # optional, hf_full: mopd.training.fsdp.shard kwargs
   "eval": {"protocol_id", "data", "data_sha256", "max_new_tokens", "checkpoints": [128, 256]},
   "test": {"protocol_id", "data", "data_sha256"},       # only with --final
   "gpus": [0, 2, ...],
   "variants": [{"name": "base", "base": true}, {"name": "...", "recipe": "configs/recipes/code/<x>.json",
                 "lora": {"rank": 64, "alpha": 128, "targets": [...]}}]}   # lora optional, default r16 q/k/v/o

Per variant: build the recipe pool (mopd.data.recipes) -> scripts/train_sft.py -> evaluate the
listed checkpoints on the dev data; both stages take a fair share of the
idle GPUs from GPUPool, so a lone variant uses every one of them. Every stage is
skipped when its output exists, so adding a variant to the sweep and re-running only trains the new
one. Runs: runs/sweeps/<sweep>/<variant>/; leaderboard: results/sweeps/<sweep>/leaderboard.{json,md}.

  python scripts/run_sweep.py configs/sweeps/code-data-v1.json [--only a b] [--final a b] [--collect]
"""
import argparse, json, os, shutil, subprocess, sys, threading, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mopd.data import recipes
from mopd.runtime.provenance import sha256
from mopd.runtime.gpus import idle_gpus
from mopd.students.registry import is_sharded
from mopd.evaluation.eval_commands import eval_commands, PY

BASE_WEIGHTS = ["model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors"]


class GPUPool:
    """Shared GPUs, handed out in a fair share of free // remaining_jobs: a variant that is the
    only one left claims every idle GPU instead of leaving them unused, and this is the honest
    version of "dynamic scheduling" -- a job's GPU count is decided fresh each time it asks (train
    start, each eval checkpoint), not fixed for the sweep's lifetime. True elastic scaling --
    growing a torchrun job that is already running, without restarting it -- is a separate, much
    larger feature (needs an elastic rendezvous and checkpoint-resume on membership change) and is
    out of scope here.

    remaining_jobs is seeded with the job count and decremented once per job via done(), not a
    count of callers *currently* inside want(): sizing the share from "how many are blocked right
    now" sounds more responsive, but is racy -- whichever caller's want() happens to run first
    would see nobody else waiting yet and grab everything, silently serializing jobs that were
    meant to run in parallel (caught by test_gpu_pool.py: N jobs arriving within a few scheduler
    ticks of each other reliably hit this, since Python thread start-up has no synchronization
    point to guarantee they all register before the first one computes its share). remaining_jobs
    only drops when a job is verifiably done, so it cannot be fooled by arrival order."""

    def __init__(self, gpus, jobs):
        self._free = list(gpus)
        self._remaining = jobs
        self._holding = 0  # grants currently out; their jobs are not competing for the free GPUs
        self._cv = threading.Condition()

    def want(self, min_gpus=1, max_gpus=None):
        """min_gpus is a floor (wait until at least this many are free); max_gpus is a ceiling on
        what this caller can use."""
        with self._cv:
            while len(self._free) < min_gpus:
                self._cv.wait()
            # split the free GPUs among the jobs that do not hold any right now: a job already
            # running on its own grant is not a competitor for what is left
            share = min(max(min_gpus, len(self._free) // max(1, self._remaining - self._holding)), len(self._free))
            if max_gpus is not None:
                share = min(share, max_gpus)
            taken, self._free = self._free[:share], self._free[share:]
            self._holding += 1
            return taken

    def give_back(self, gpus):
        with self._cv:
            self._free.extend(gpus)
            self._holding -= 1
            self._cv.notify_all()

    def done(self):
        """Call once per job (train + all its evals), success or failure, when it will never call
        want() again -- shrinks the denominator so whoever is left gets a bigger share."""
        with self._cv:
            self._remaining -= 1
            self._cv.notify_all()


class Sweep:
    def __init__(self, path):
        self.spec = json.loads(Path(path).read_text())
        self.name = self.spec["name"]
        self.runs = ROOT / "runs/sweeps" / self.name
        self.results = ROOT / "results/sweeps" / self.name
        self.results.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def log(self, **fields):
        line = json.dumps({"time": time.strftime("%H:%M:%S"), **fields})
        with self.lock, (self.results / "events.jsonl").open("a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    # ------------------------------------------------------------ stages
    def config(self, v, manifest):
        c, ev = self.spec["controls"], self.spec["eval"]
        pool = ROOT / "data/recipes" / manifest["name"]
        student = Path(c["student"])
        pins = {str(pool / "train.jsonl"): manifest["train_sha256"], str(pool / "dev.jsonl"): manifest["dev_sha256"],
                ev["data"]: ev["data_sha256"], **{str(student / w): sha256(student / w) for w in BASE_WEIGHTS}}
        micro = manifest["recipe"].get("micro_batch_size", 2)
        # student_device is cuda:0 regardless of GPU count: CUDA_VISIBLE_DEVICES always remaps the
        # assigned GPU(s) to start at index 0 (single-GPU convention used throughout this codebase).
        # Under torchrun with world_size > 1, train_sft.py derives the real per-rank device from
        # LOCAL_RANK instead and ignores this field; it only takes effect for a single-GPU run.
        # optional training-system knobs, same for every variant of a sweep
        system = {k: c[k] for k in ("fsdp", "gradient_checkpointing") if k in c}
        return {"student": c["student"], "student_backend": c.get("student_backend", "hf_lora"), **system,
                "student_prompt_format": "plain_qa_v1",
                "seed": c["seed"], "steps": c["steps"], "lr": c["lr"], "lr_schedule": c["lr_schedule"],
                "micro_batch_size": micro, "save_every": c["save_every"], "max_seq_len": manifest["recipe"]["max_total_tokens"],
                "output": str(self.runs / v["name"]), "data_pins": pins, "training_method": "sft", "student_device": "cuda:0",
                **({"lora": v["lora"]} if "lora" in v else {}),  # e.g. {"rank": 64, "alpha": 128, "targets": [...]}
                "sweep": {"name": self.name, "variant": v["name"], "recipe": v["recipe"], "recipe_version": manifest["version"]},
                "domains": {"code": {"train_file": str(pool / "train.jsonl"), "holdout_files": [str(pool / "dev.jsonl"), ev["data"]],
                                     "count": c["samples_per_step"], "weight": 1.0}}}

    def trained(self, v):
        status = self.runs / v["name"] / "status.json"
        return status.exists() and json.loads(status.read_text()).get("state") == "complete"

    def eval_dir(self, v, ckpt, kind="eval"):
        protocol = self.spec[kind]["protocol_id"]
        if v.get("base"):
            return self.results / "base" / protocol
        return self.runs / v["name"] / "evaluations" / f"checkpoint-{ckpt:04d}" / protocol

    def evaluate(self, v, ckpt, pool, log, kind="eval"):
        """On however many GPUs the pool grants right now; generate_benchmark_vllm.py hands rows to
        whichever of them has room."""
        ev = {**self.spec["eval"], **self.spec.get(kind, {})}
        out = self.eval_dir(v, ckpt, kind)
        if (out / "summary.json").exists():
            return True
        gpus = pool.want()
        try:
            self.log(variant=v["name"], stage=f"{kind}_start", checkpoint=ckpt, gpus=gpus)
            common = dict(data=ev["data"], protocol_id=ev["protocol_id"], max_new_tokens=ev["max_new_tokens"], device=gpus)
            if v.get("base"):
                cmds = eval_commands(**common, out=out, model=self.spec["controls"]["student"], model_version_id="base:Qwen3-4B-Base")
            else:
                cmds = eval_commands(**common, model_version=str(self.runs / v["name"] / "model-version.json"), checkpoint=f"checkpoint-{ckpt:04d}")
            return all(subprocess.run(cmd, cwd=ROOT, env={**os.environ, **extra}, stdout=log, stderr=subprocess.STDOUT).returncode == 0
                       for cmd, extra in cmds)
        finally:
            pool.give_back(gpus)

    def train(self, v, manifest, pool, log):
        """Build the SFT config and launch train_sft.py across however many GPUs the pool grants
        right now, via torchrun (nproc_per_node=1 behaves identically to a plain invocation, so
        this is the one code path for every GPU count -- see train_sft.py's module docstring for
        why multi-GPU is a manual gradient all-reduce rather than DistributedDataParallel)."""
        run_dir = self.runs / v["name"]
        if run_dir.exists():
            # train_sft.py refuses to resume (mkdir exist_ok=False): a prior killed/crashed
            # attempt leaves a directory train_sft.py won't touch, so clear it before retrying.
            shutil.rmtree(run_dir)
            self.log(variant=v["name"], stage="cleanup", removed=str(run_dir))
        cfg_path = self.results / "configs" / f"{v['name']}.json"
        cfg_path.parent.mkdir(exist_ok=True)
        cfg_path.write_text(json.dumps(self.config(v, manifest), indent=1))
        gpus = pool.want(min_gpus=4 if is_sharded(self.config(v, manifest)) else 1)  # full fine-tuning: see train_sft
        try:
            self.log(variant=v["name"], stage="train_start", gpus=gpus)
            start = time.time()
            cmd = [PY, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={len(gpus)}",
                   "scripts/train_sft.py", "--config", str(cfg_path)]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus))}
            rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        finally:
            pool.give_back(gpus)
        self.log(variant=v["name"], stage="train", ok=rc == 0, gpus=len(gpus), minutes=round((time.time() - start) / 60, 1))
        return rc == 0

    def run_variant(self, v, pool, final):
        try:
            return self._run_variant(v, pool, final)
        finally:
            pool.done()

    def _run_variant(self, v, pool, final):
        logdir = self.results / "logs"; logdir.mkdir(exist_ok=True)
        with (logdir / f"{v['name']}.log").open("a") as log:
            if not v.get("base"):
                manifest = recipes.build(ROOT / v["recipe"])
                self.log(variant=v["name"], stage="recipe", rows=manifest["train_rows"], version=manifest["version"])
                if not self.trained(v) and not self.train(v, manifest, pool, log):
                    return False
            ckpts = [0] if v.get("base") else self.spec["eval"]["checkpoints"]
            for ck in ckpts:
                ok = self.evaluate(v, ck, pool, log)
                self.log(variant=v["name"], stage="eval", checkpoint=ck, ok=ok)
                if not ok:
                    return False
            if final:
                ok = self.evaluate(v, ckpts[-1], pool, log, kind="test")
                self.log(variant=v["name"], stage="test", checkpoint=ckpts[-1], ok=ok)
        return True

    def run(self, only, final):
        if sha256(self.spec["eval"]["data"]) != self.spec["eval"]["data_sha256"]:
            raise SystemExit("dev benchmark digest differs from the sweep pin")
        todo = [v for v in self.spec["variants"] if not only or v["name"] in only]
        # Live check, not a fixed allow-list: this server is shared, and another team's job can
        # occupy any GPU (including one we've always treated as "ours") with no warning in our own
        # config. Only GPUs idle right at launch go in the pool; a GPU that was free then but gets
        # claimed by someone else mid-sweep is a race this preflight-style check cannot catch, same
        # as run_experiment.py's.
        idle = idle_gpus(self.spec["gpus"])
        busy = [g for g in self.spec["gpus"] if g not in idle]
        if busy:
            self.log(stage="gpus_excluded", busy=busy, reason="in use by another process")
        pool = GPUPool(idle, jobs=len(todo))
        failed = []

        def worker(v):
            try:
                ok = self.run_variant(v, pool, final and v["name"] in final)
            except Exception as e:  # a broken recipe must not stop the other variants
                self.log(variant=v["name"], stage="error", error=repr(e)); ok = False
            if not ok:
                failed.append(v["name"])

        threads = [threading.Thread(target=worker, args=(v,)) for v in todo]
        for t in threads: t.start()
        for t in threads: t.join()
        self.collect()
        if failed:
            raise SystemExit(f"failed variants: {failed}")

    # ------------------------------------------------------------ leaderboard
    def scores(self, out, data):
        cases = [json.loads(l) for l in (out / "cases.jsonl").read_text().splitlines()]
        res = {}
        for name, keep in (("all", lambda m: True), ("easy", lambda m: m["difficulty"] == "easy"),
                           ("medium", lambda m: m["difficulty"] == "medium"), ("hard", lambda m: m["difficulty"] == "hard")):
            rows = [c for c in cases if keep(data[c["id"]]["metadata"])]
            res[name] = round(100 * sum(c["correct"] for c in rows) / len(rows), 1)
        res["lenient"] = round(100 * sum(c.get("correct_lenient", c["correct"]) for c in cases) / len(cases), 1)
        res["truncated"] = sum(c.get("truncated", False) for c in cases)
        return res

    def collect(self):
        board = []
        for kind in ("eval", "test"):
            if kind not in self.spec:
                continue
            data = {json.loads(l)["id"]: json.loads(l) for l in Path(self.spec[kind]["data"]).read_text().splitlines()}
            for v in self.spec["variants"]:
                for ck in ([0] if v.get("base") else self.spec["eval"]["checkpoints"]):
                    out = self.eval_dir(v, ck, kind)
                    if (out / "cases.jsonl").exists():
                        m = json.loads((ROOT / "data/recipes" / json.loads((ROOT / v["recipe"]).read_text())["name"] / "manifest.json").read_text()) if not v.get("base") else None
                        board.append({"set": kind, "variant": v["name"], "checkpoint": ck, **self.scores(out, data),
                                      "recipe_version": m and m["version"], "train_rows": m and m["train_rows"]})
        (self.results / "leaderboard.json").write_text(json.dumps(board, indent=1))
        base = {b["set"]: b["all"] for b in board if b["variant"] == "base"}
        lines = [f"# {self.name}", "", self.spec.get("description", ""), "",
                 "| set | variant | ckpt | all | Δbase | easy | medium | hard | lenient | truncated |",
                 "|---|---|---|---|---|---|---|---|---|---|"]
        for b in sorted(board, key=lambda b: (b["set"], -b["all"])):
            d = f"{b['all'] - base[b['set']]:+.1f}" if b["set"] in base else ""
            lines.append(f"| {b['set']} | {b['variant']} | {b['checkpoint']} | {b['all']} | {d} | {b['easy']} | {b['medium']} | {b['hard']} | {b['lenient']} | {b['truncated']} |")
        (self.results / "leaderboard.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--final", nargs="*", default=[])
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--gpus", type=int, nargs="*", help="override spec[\"gpus\"] (e.g. to rerun failed variants on GPUs not busy with another in-flight sweep run)")
    a = ap.parse_args()
    s = Sweep(a.sweep)
    if a.gpus:
        s.spec["gpus"] = a.gpus
    if a.collect:
        s.collect()
    else:
        s.run(a.only, a.final)


if __name__ == "__main__":
    main()
