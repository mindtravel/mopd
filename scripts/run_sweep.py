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
                 "lora": {"rank": 64, "alpha": 128, "targets": [...]}}]}   # lora optional, default r16 q/k/v/o

Per variant: build the recipe pool (mopd.data.recipes) -> scripts/train_sft.py -> evaluate the listed
checkpoints on the dev data. Jobs run through mopd.runtime.scheduler (one DAG over the idle GPUs,
fair dynamic shares, a bandwidth gate before sharded trainings). Every stage is skipped when its
output exists, so adding a variant to the sweep and re-running only trains the new one.
Runs: runs/sweeps/<sweep>/<variant>/; leaderboard: results/sweeps/<sweep>/leaderboard.{json,md}.

  python scripts/run_sweep.py configs/sweeps/code-data-v1.json [--only a b] [--final a b] [--collect]
"""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mopd.data import recipes
from mopd.runtime.provenance import sha256
from mopd.runtime.gpus import idle_gpus
from mopd.runtime.scheduler import Scheduler
from mopd.students.registry import is_sharded
from mopd.evaluation.eval_commands import eval_commands, PY

BASE_WEIGHTS = ["model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors"]


class Sweep:
    def __init__(self, path):
        self.spec = json.loads(Path(path).read_text())
        self.name = self.spec["name"]
        self.runs = ROOT / "runs/sweeps" / self.name
        self.results = ROOT / "results/sweeps" / self.name
        self.results.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ stages
    def config(self, v, manifest):
        c, ev = self.spec["controls"], self.spec["eval"]
        pool = ROOT / "data/recipes" / manifest["name"]
        student = Path(c["student"])
        pins = {str(pool / "train.jsonl"): manifest["train_sha256"], str(pool / "dev.jsonl"): manifest["dev_sha256"],
                ev["data"]: ev["data_sha256"], **{str(student / w): sha256(student / w) for w in BASE_WEIGHTS}}
        micro = manifest["recipe"].get("micro_batch_size", 2)
        budget = manifest["recipe"].get("micro_batch_tokens")  # padded-token cap per micro-batch (mopd.training.microbatch)
        # student_device is cuda:0 regardless of GPU count: CUDA_VISIBLE_DEVICES always remaps the
        # assigned GPU(s) to start at index 0. Under torchrun with world_size > 1, train_sft.py derives
        # the real per-rank device from LOCAL_RANK instead and ignores this field.
        system = {k: c[k] for k in ("fsdp", "gradient_checkpointing") if k in c}  # same for every variant
        return {"student": c["student"], "student_backend": c.get("student_backend", "hf_lora"), **system,
                "student_prompt_format": c.get("student_prompt_format", "plain_qa_v1"),
                "seed": c["seed"], "steps": c["steps"], "lr": c["lr"], "lr_schedule": c["lr_schedule"],
                "micro_batch_size": micro, **({"micro_batch_tokens": budget} if budget else {}),
                "save_every": c["save_every"], "max_seq_len": manifest["recipe"]["max_total_tokens"],
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

    def shell(self, cmds, log):
        with log.open("a") as f:
            for cmd, extra in cmds:
                f.write(f"\n$ {' '.join(map(str, cmd))}\n"); f.flush()
                subprocess.run(cmd, cwd=ROOT, env={**os.environ, **extra}, stdout=f, stderr=subprocess.STDOUT, check=True)

    def evaluate(self, v, ckpt, gpus, log, kind="eval"):
        """On the granted GPUs; generate_benchmark_vllm.py hands rows to whichever of them has room."""
        ev = {**self.spec["eval"], **self.spec.get(kind, {})}
        out = self.eval_dir(v, ckpt, kind)
        common = dict(data=ev["data"], protocol_id=ev["protocol_id"], max_new_tokens=ev["max_new_tokens"], device=gpus)
        if v.get("base"):
            cmds = eval_commands(**common, out=out, model=self.spec["controls"]["student"], model_version_id="base:Qwen3-4B-Base",
                                 student_prompt_format=self.spec["controls"].get("student_prompt_format", "plain_qa_v1"))
        else:
            cmds = eval_commands(**common, model_version=str(self.runs / v["name"] / "model-version.json"), checkpoint=f"checkpoint-{ckpt:04d}")
        self.shell(cmds, log)

    def train(self, v, manifest, gpus, log):
        """scripts/train_sft.py across the granted GPUs via torchrun (nproc_per_node=1 is a plain run)."""
        run_dir = self.runs / v["name"]
        if run_dir.exists():  # a prior killed/crashed attempt; train_sft.py refuses to resume into it
            shutil.rmtree(run_dir)
        cfg_path = self.results / "configs" / f"{v['name']}.json"
        cfg_path.parent.mkdir(exist_ok=True)
        cfg_path.write_text(json.dumps(self.config(v, manifest), indent=1))
        cmd = [PY, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={len(gpus)}",
               "scripts/train_sft.py", "--config", str(cfg_path)]
        self.shell([(cmd, {"CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus))})], log)

    # ------------------------------------------------------------ run
    def jobs(self, scheduler, todo, final):
        logdir = self.results / "logs"; logdir.mkdir(exist_ok=True)
        for v in todo:
            log = logdir / f"{v['name']}.log"
            ckpts = [0] if v.get("base") else self.spec["eval"]["checkpoints"]
            deps = []
            if not v.get("base"):
                manifest = recipes.build(ROOT / v["recipe"])  # cached: built once up front in run()
                sharded = is_sharded(self.config(v, manifest))
                scheduler.add({"id": f"train:{v['name']}", "order": 1, "done": lambda v=v: self.trained(v),
                               "gpus": {"min": 4 if sharded else 1}, "gate": "collectives" if sharded else None,
                               "run": lambda gpus, v=v, m=manifest, log=log: self.train(v, m, gpus, log)})
                deps = [f"train:{v['name']}"]
            for ck in ckpts:
                scheduler.add({"id": f"eval:{v['name']}:{ck}", "deps": deps, "order": 5, "gpus": {"min": 1},
                               "done": lambda v=v, ck=ck: (self.eval_dir(v, ck) / "summary.json").exists(),
                               "run": lambda gpus, v=v, ck=ck, log=log: self.evaluate(v, ck, gpus, log)})
            if final:
                scheduler.add({"id": f"test:{v['name']}", "deps": deps, "order": 6, "gpus": {"min": 1},
                               "done": lambda v=v, ck=ckpts[-1]: (self.eval_dir(v, ck, "test") / "summary.json").exists(),
                               "run": lambda gpus, v=v, ck=ckpts[-1], log=log: self.evaluate(v, ck, gpus, log, kind="test")})

    def run(self, only, final):
        if sha256(self.spec["eval"]["data"]) != self.spec["eval"]["data_sha256"]:
            raise SystemExit("dev benchmark digest differs from the sweep pin")
        todo = [v for v in self.spec["variants"] if not only or v["name"] in only]
        # Build every recipe up front, one at a time: concurrent first imports of transformers from
        # several threads raced inside its lazy module ("cannot import name 'AutoTokenizer'").
        for v in todo:
            if not v.get("base"):
                m = recipes.build(ROOT / v["recipe"])
                print(json.dumps({"variant": v["name"], "stage": "recipe", "rows": m["train_rows"], "version": m["version"]}), flush=True)
        # Live check, not a fixed allow-list: the server is shared and another team's job can occupy
        # any GPU. Only GPUs idle right at launch go in the pool.
        idle = idle_gpus(self.spec["gpus"])
        scheduler = Scheduler(idle, self.results, name="sweep")
        busy = [g for g in self.spec["gpus"] if g not in idle]
        if busy:
            scheduler.log("gpus_excluded", busy=busy, reason="in use by another process")
        self.jobs(scheduler, todo, final)
        state = scheduler.run()
        self.collect()
        failed = sorted({k.split(":")[1] for k, s in state.items() if s["state"] in ("failed", "blocked")})
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
    ap.add_argument("--gpus", type=int, nargs="*", help="override spec[\"gpus\"]")
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
