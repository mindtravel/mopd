"""Data-ablation sweep: one SFT expert per recipe, scored on a dev benchmark, collected in a leaderboard.

  configs/sweeps/<name>.json
  {"name", "description",
   "controls": {"student", "seed", "steps", "samples_per_step", "lr", "lr_schedule",
                "save_every",                      # max_seq_len and micro_batch_size come from each recipe
                "student_backend",                 # optional: "hf_lora" (default) or "hf_full"
                "student_prompt_format",           # optional: plain_qa_v1 (default) or plain_qa_think_v1
                "fsdp", "gradient_checkpointing"}, # optional, hf_full: mopd.training.fsdp.shard kwargs
   "eval": {"protocol_id", "data", "data_sha256", "max_new_tokens", "checkpoints": [128, 256]},
   "test": {"protocol_id", "data", "data_sha256"},       # only with --final
   "gpus": [0, 2, ...],
   "variants": [{"name": "base", "base": true}, {"name": "...", "recipe": "configs/recipes/code/<x>.json",
                 "lora": {"rank": 64, "alpha": 128, "targets": [...]},   # optional, default r16 q/k/v/o
                 "controls": {"lr": 5e-06},                               # optional per-variant control overrides
                 "checkpoints": [64, 128, 256]}]}                         # optional, default eval.checkpoints

Per variant: build the recipe pool (mopd.data.recipes) -> scripts/train_sft.py -> evaluate the listed
checkpoints on the dev data. Jobs run through mopd.runtime.scheduler (one DAG over the idle GPUs,
fair dynamic shares, a bandwidth gate before sharded trainings). Every stage is skipped when its
output exists, so adding a variant to the sweep and re-running only trains the new one.
Runs: runs/sweeps/<sweep>/<variant>/; leaderboard: results/sweeps/<sweep>/leaderboard.{json,md}.

  python scripts/run_sweep.py configs/sweeps/code-data-v1.json [--only a b] [--final a b] [--collect]
"""

import argparse, json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mopd.data import recipes
from mopd.runtime.provenance import sha256
from mopd.runtime.scheduler import Scheduler, usable_gpus
from mopd.evaluation import jobs, report
from mopd.training import controls as training_controls
from mopd.evaluation.eval_commands import PY

BASE_WEIGHTS = [
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
]


class Sweep:
    def __init__(self, path):
        self.spec = json.loads(Path(path).read_text())
        self.name = self.spec["name"]
        self.runs = ROOT / "runs/sweeps" / self.name
        self.results = ROOT / "results/sweeps" / self.name
        self.results.mkdir(parents=True, exist_ok=True)
        training_controls.check(self.spec["controls"], path)
        for v in self.spec["variants"]:
            training_controls.check(v.get("controls", {}), f"{path}: variant {v['name']}")

    # ------------------------------------------------------------ stages
    def config(self, v, manifest):
        c, ev = {**self.spec["controls"], **v.get("controls", {})}, self.spec["eval"]  # per-variant overrides, e.g. lr
        pool = ROOT / "data/recipes" / manifest["name"]
        student = Path(c["student"])
        pins = {
            str(pool / "train.jsonl"): manifest["train_sha256"],
            str(pool / "dev.jsonl"): manifest["dev_sha256"],
            ev["data"]: ev["data_sha256"],
            **{str(student / w): sha256(student / w) for w in BASE_WEIGHTS},
        }
        micro = manifest["recipe"].get("micro_batch_size", 2)
        budget = manifest["recipe"].get(
            "micro_batch_tokens"
        )  # padded-token cap per micro-batch (mopd.training.microbatch)
        # student_device is cuda:0 regardless of GPU count: CUDA_VISIBLE_DEVICES always remaps the
        # assigned GPU(s) to start at index 0. Under torchrun with world_size > 1, train_sft.py derives
        # the real per-rank device from LOCAL_RANK instead and ignores this field.
        return {
            **training_controls.training_fields(c, micro_batch_size=micro),
            **({"micro_batch_tokens": budget} if budget else {}),
            **training_controls.system_fields(c),
            **({"lora": v["lora"]} if "lora" in v else {}),  # e.g. {"rank": 64, "alpha": 128, "targets": [...]}
            "max_seq_len": manifest["recipe"]["max_total_tokens"],
            "output": str(self.runs / v["name"]),
            "data_pins": pins,
            "training_method": "sft",
            "student_device": "cuda:0",
            "sweep": {
                "name": self.name,
                "variant": v["name"],
                "recipe": v["recipe"],
                "recipe_version": manifest["version"],
            },
            "domains": {
                "code": {
                    "train_file": str(pool / "train.jsonl"),
                    "holdout_files": [str(pool / "dev.jsonl"), ev["data"]],
                    "count": c["samples_per_step"],
                    "weight": 1.0,
                }
            },
        }

    def checkpoints(self, v):
        """Checkpoints to evaluate: the variant's own list ("checkpoints") or the sweep's."""
        return [0] if v.get("base") else v.get("checkpoints", self.spec["eval"]["checkpoints"])

    def trained(self, v):
        status = self.runs / v["name"] / "status.json"
        return status.exists() and json.loads(status.read_text()).get("state") == "complete"

    def eval_dir(self, v, ckpt, kind="eval"):
        protocol = self.spec[kind]["protocol_id"]
        if v.get("base"):
            return self.results / "base" / protocol
        return self.runs / v["name"] / "evaluations" / f"checkpoint-{ckpt:04d}" / protocol

    def model_args(self, v, ckpt):
        """eval_commands' model arguments: the base model, or a run's checkpoint."""
        if v.get("base"):
            c = self.spec["controls"]
            return dict(
                model=c["student"],
                model_version_id="base:Qwen3-4B-Base",
                student_prompt_format=c.get("student_prompt_format", "plain_qa_v1"),
            )
        return dict(
            model_version=str(self.runs / v["name"] / "model-version.json"), checkpoint=f"checkpoint-{ckpt:04d}"
        )

    def train(self, v, manifest, gpus, log):
        """scripts/train_sft.py across the granted GPUs via torchrun (nproc_per_node=1 is a plain run)."""
        run_dir = self.runs / v["name"]
        if run_dir.exists():  # a prior killed/crashed attempt; train_sft.py refuses to resume into it
            shutil.rmtree(run_dir)
        cfg_path = self.results / "configs" / f"{v['name']}.json"
        cfg_path.parent.mkdir(exist_ok=True)
        cfg_path.write_text(json.dumps(self.config(v, manifest), indent=1))
        cmd = [
            PY,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={len(gpus)}",
            "scripts/train_sft.py",
            "--config",
            str(cfg_path),
        ]
        jobs.shell([(cmd, {"CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus))})], log)

    # ------------------------------------------------------------ run
    def add_jobs(self, scheduler, todo, final):
        logdir = self.results / "logs"
        logdir.mkdir(exist_ok=True)
        for v in todo:
            log = logdir / f"{v['name']}.log"
            ckpts = self.checkpoints(v)
            deps = []
            if not v.get("base") and not self.trained(v):  # already-trained variants: evaluation only
                manifest = recipes.build(ROOT / v["recipe"])  # cached: built once up front in run()
                scheduler.add(
                    {
                        "id": f"train:{v['name']}",
                        "order": 1,
                        "done": lambda v=v: self.trained(v),
                        "run": lambda gpus, v=v, m=manifest, log=log: self.train(v, m, gpus, log),
                        **jobs.training_resources(self.config(v, manifest)),
                    }
                )
                deps = [f"train:{v['name']}"]
            for kind, cks in (("eval", ckpts), ("test", ckpts[-1:] if final else [])):
                ev = {**self.spec["eval"], **self.spec.get(kind, {})}
                for ck in cks:
                    for job in jobs.eval_jobs(
                        f"{kind}:{v['name']}:{ck}",
                        out=self.eval_dir(v, ck, kind),
                        data=ev["data"],
                        protocol_id=ev["protocol_id"],
                        max_new_tokens=ev["max_new_tokens"],
                        log=log,
                        deps=deps,
                        order=5 if kind == "eval" else 6,
                        **self.model_args(v, ck),
                    ):
                        scheduler.add(job)

    def run(self, only, final):
        if sha256(self.spec["eval"]["data"]) != self.spec["eval"]["data_sha256"]:
            raise SystemExit("dev benchmark digest differs from the sweep pin")
        todo = [v for v in self.spec["variants"] if not only or v["name"] in only]
        # Build every recipe up front, one at a time: concurrent first imports of transformers from
        # several threads raced inside its lazy module ("cannot import name 'AutoTokenizer'").
        for v in todo:
            if not v.get("base") and not self.trained(v):
                m = recipes.build(ROOT / v["recipe"])
                print(
                    json.dumps(
                        {"variant": v["name"], "stage": "recipe", "rows": m["train_rows"], "version": m["version"]}
                    ),
                    flush=True,
                )
        # Live check, not a fixed allow-list: the server is shared and another team's job can occupy
        # any GPU. Only GPUs idle right at launch go in the pool.
        idle = usable_gpus(self.spec["gpus"])  # not another team's; ours under a lease are waited for
        scheduler = Scheduler(idle, self.results, name="sweep")
        busy = [g for g in self.spec["gpus"] if g not in idle]
        if busy:
            scheduler.log("gpus_excluded", busy=busy, reason="in use by another process")
        self.add_jobs(scheduler, todo, final)
        state = scheduler.run()
        self.collect()
        failed = sorted({k.split(":")[1] for k, s in state.items() if s["state"] in ("failed", "blocked")})
        if failed:
            raise SystemExit(f"failed variants: {failed}")

    # ------------------------------------------------------------ leaderboard
    def collect(self):
        board = []
        for kind in ("eval", "test"):
            if kind not in self.spec:
                continue
            data = report.load_rows(self.spec[kind]["data"])
            for v in self.spec["variants"]:
                for ck in self.checkpoints(v):
                    out = self.eval_dir(v, ck, kind)
                    if (out / "cases.jsonl").exists():
                        m = (
                            json.loads(
                                (
                                    ROOT
                                    / "data/recipes"
                                    / json.loads((ROOT / v["recipe"]).read_text())["name"]
                                    / "manifest.json"
                                ).read_text()
                            )
                            if not v.get("base")
                            else None
                        )
                        board.append(
                            {
                                "set": kind,
                                "variant": v["name"],
                                "checkpoint": ck,
                                **report.lcb_breakdown(out / "cases.jsonl", data),
                                "recipe_version": m and m["version"],
                                "train_rows": m and m["train_rows"],
                            }
                        )
        (self.results / "leaderboard.json").write_text(json.dumps(board, indent=1))
        base = {b["set"]: b["all"] for b in board if b["variant"] == "base"}
        lines = [
            f"# {self.name}",
            "",
            self.spec.get("description", ""),
            "",
            "| set | variant | ckpt | all | Δbase | functional | stdin | easy | medium | hard | truncated |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for b in sorted(board, key=lambda b: (b["set"], -b["all"])):
            d = f"{b['all'] - base[b['set']]:+.1f}" if b["set"] in base else ""
            lines.append(
                f"| {b['set']} | {b['variant']} | {b['checkpoint']} | {b['all']} | {d} | {b['functional']} | {b['stdin']}"
                f" | {b['easy']} | {b['medium']} | {b['hard']} | {b['truncated']} |"
            )
        (self.results / "leaderboard.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--final", nargs="*", default=[])
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--gpus", type=int, nargs="*", help='override spec["gpus"]')
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
