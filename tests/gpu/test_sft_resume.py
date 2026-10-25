"""GPU check of preemptible full-parameter SFT (TrainRun.save_resume / restore): a 6-step run stopped on
request after step 3 (the pool's $MOPD_POOL_STOP) must exit 75 with resume/ saved, and resuming on 2
GPUs with offload -- another GPU count, so the state is re-sharded -- must continue with the losses of
an uninterrupted 4-GPU run (equal up to the summation order of another world size).

  launched by the pool on 4 GPUs:  python tests/gpu/test_sft_resume.py
"""

import json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mopd import paths
from mopd.study import resolve, spec

STUDY = ROOT / "configs/studies/full-sft-lr.yaml"
OUT = paths.ROOT / "runs/scratch/sft-resume-check"


def config(name, **over):
    st = spec.load(STUDY)
    fields = dict(spec.expand(st))["code-lr2e-06"]
    c = resolve.run_config(st, name, fields, [], str(STUDY), "check", resolve.registry())
    c = {**c, "steps": 6, "save_every": 1000, "output": str(OUT / name), **over}
    c.pop("experiment", None)  # a check, not a study run (provenance does not tie it to the spec)
    p = OUT / f"{name}.json"
    p.write_text(json.dumps(c, indent=1))
    return p


def train(cfg, gpus, env=None):
    return subprocess.run([str(paths.TRAIN_PY), "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={gpus}",
                           "scripts/train_sft.py", "--config", str(cfg)], cwd=ROOT,
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, range(gpus))), **(env or {})}).returncode


def losses(name):
    return {m["step"]: m["loss"] for m in map(json.loads, (OUT / name / "metrics.jsonl").read_text().splitlines())}


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    assert train(config("whole"), 4) == 0
    stop = OUT / "stop"
    # the stop file appears once the run has logged step 3 (as the pool would create it mid-run)
    import threading, time

    def watch():
        f = OUT / "cut/metrics.jsonl"
        while not (f.exists() and any(json.loads(l)["step"] >= 3 for l in f.read_text().splitlines() if l.strip())):
            time.sleep(0.5)
        stop.write_text("stop")

    threading.Thread(target=watch, daemon=True).start()
    rc = train(config("cut"), 4, {"MOPD_POOL_STOP": str(stop)})
    meta = json.loads((OUT / "cut/resume/meta.json").read_text())
    assert rc == 75, rc
    stop.unlink()
    resumed_cfg = config("cut", fsdp={"reshard_after_forward": True, "offload": True})
    assert train(resumed_cfg, 2) == 0
    a, b = losses("whole"), losses("cut")
    print(json.dumps({"stopped_after": meta["step"], "whole": a, "resumed": b}))
    assert sorted(b) == list(range(1, 7)), sorted(b)
    for s in range(meta["step"] + 1, 7):
        assert abs(a[s] - b[s]) < 2e-3 * max(1.0, abs(a[s])), (s, a[s], b[s])
    print("sft resume ok")


if __name__ == "__main__":
    main()
