"""Data-ablation sweep: one SFT expert per recipe, scored on a dev benchmark, collected in a leaderboard.

  configs/sweeps/<name>.json
  {"name", "description",
   "controls": {"student", "seed", "steps", "samples_per_step", "lr", "lr_schedule",
                "save_every"},                     # max_seq_len and micro_batch_size come from each recipe
   "eval": {"protocol_id", "data", "data_sha256", "max_new_tokens", "checkpoints": [128, 256]},
   "test": {"protocol_id", "data", "data_sha256"},       # only with --final
   "gpus": [0, 2, ...],
   "variants": [{"name": "base", "base": true}, {"name": "...", "recipe": "configs/recipes/code/<x>.json"}]}

Per variant, on one free GPU: build the recipe pool (mopd.data.recipes) -> scripts/train_sft.py ->
evaluate the listed checkpoints on the dev data. Every stage is skipped when its output exists, so
adding a variant to the sweep and re-running only trains the new one. Runs:
runs/sweeps/<sweep>/<variant>/; leaderboard: results/sweeps/<sweep>/leaderboard.{json,md}.

  python scripts/run_sweep.py configs/sweeps/code-data-v1.json [--only a b] [--final a b] [--collect]
"""
import argparse, json, queue, subprocess, sys, threading, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mopd.data import recipes
from mopd.runtime.provenance import sha256
from run_experiment import EVAL_SCHEDULING, PY

BASE_WEIGHTS = ["model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors"]


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
    def config(self, v, manifest, gpu):
        c, ev = self.spec["controls"], self.spec["eval"]
        pool = ROOT / "data/recipes" / manifest["name"]
        student = Path(c["student"])
        pins = {str(pool / "train.jsonl"): manifest["train_sha256"], str(pool / "dev.jsonl"): manifest["dev_sha256"],
                ev["data"]: ev["data_sha256"], **{str(student / w): sha256(student / w) for w in BASE_WEIGHTS}}
        micro = manifest["recipe"].get("micro_batch_size", 2)
        return {"student": c["student"], "student_backend": "hf_lora", "student_prompt_format": "plain_qa_v1",
                "seed": c["seed"], "steps": c["steps"], "lr": c["lr"], "lr_schedule": c["lr_schedule"],
                "micro_batch_size": micro, "save_every": c["save_every"], "max_seq_len": manifest["recipe"]["max_total_tokens"],
                "output": str(self.runs / v["name"]), "data_pins": pins, "training_method": "sft", "student_device": "cuda:0",
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

    def evaluate(self, v, ckpt, gpu, log, kind="eval"):
        ev = {**self.spec["eval"], **self.spec.get(kind, {})}
        out = self.eval_dir(v, ckpt, kind)
        if (out / "summary.json").exists():
            return True
        common = ["--data", ev["data"], "--protocol-id", ev["protocol_id"], "--max-new-tokens", str(ev["max_new_tokens"])]
        if v.get("base"):
            cmds = [[str(ROOT / "mopd_infer/bin/python"), "scripts/generate_benchmark_vllm.py", "--model", self.spec["controls"]["student"],
                     "--out", str(out), "--device", str(gpu), "--model-version-id", "base:Qwen3-4B-Base",
                     "--student-prompt-format", "plain_qa_v1", *common, *EVAL_SCHEDULING],
                    [PY, "scripts/grade_benchmark_outputs.py", "--data", ev["data"], "--out", str(out)]]
        else:
            cmds = [[PY, "scripts/evaluate_model_version.py", "--model-version", str(self.runs / v["name"] / "model-version.json"),
                     "--checkpoint", f"checkpoint-{ckpt:04d}", "--device", str(gpu), *common, *EVAL_SCHEDULING]]
        return all(subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode == 0 for cmd in cmds)

    def run_variant(self, v, gpu, final):
        logdir = self.results / "logs"; logdir.mkdir(exist_ok=True)
        with (logdir / f"{v['name']}.log").open("a") as log:
            if not v.get("base"):
                manifest = recipes.build(ROOT / v["recipe"])
                self.log(variant=v["name"], stage="recipe", rows=manifest["train_rows"], version=manifest["version"])
                if not self.trained(v):
                    cfg_path = self.results / "configs" / f"{v['name']}.json"
                    cfg_path.parent.mkdir(exist_ok=True)
                    cfg_path.write_text(json.dumps(self.config(v, manifest, gpu), indent=1))
                    start = time.time()
                    rc = subprocess.run([PY, "scripts/train_sft.py", "--config", str(cfg_path)], cwd=ROOT, stdout=log,
                                        stderr=subprocess.STDOUT, env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": str(gpu)}).returncode
                    self.log(variant=v["name"], stage="train", ok=rc == 0, minutes=round((time.time() - start) / 60, 1))
                    if rc:
                        return False
            ckpts = [0] if v.get("base") else self.spec["eval"]["checkpoints"]
            for ck in ckpts:
                ok = self.evaluate(v, ck, gpu, log)
                self.log(variant=v["name"], stage="eval", checkpoint=ck, ok=ok)
                if not ok:
                    return False
            if final:
                ok = self.evaluate(v, ckpts[-1], gpu, log, kind="test")
                self.log(variant=v["name"], stage="test", checkpoint=ckpts[-1], ok=ok)
        return True

    def run(self, only, final):
        if sha256(self.spec["eval"]["data"]) != self.spec["eval"]["data_sha256"]:
            raise SystemExit("dev benchmark digest differs from the sweep pin")
        todo = queue.Queue()
        for v in self.spec["variants"]:
            if not only or v["name"] in only:
                todo.put(v)
        failed = []

        def worker(gpu):
            while True:
                try:
                    v = todo.get_nowait()
                except queue.Empty:
                    return
                try:
                    ok = self.run_variant(v, gpu, final and v["name"] in final)
                except Exception as e:  # a broken recipe must not stop the other variants
                    self.log(variant=v["name"], stage="error", error=repr(e)); ok = False
                if not ok:
                    failed.append(v["name"])

        threads = [threading.Thread(target=worker, args=(g,)) for g in self.spec["gpus"]]
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
    a = ap.parse_args()
    s = Sweep(a.sweep)
    if a.collect:
        s.collect()
    else:
        s.run(a.only, a.final)


if __name__ == "__main__":
    main()
