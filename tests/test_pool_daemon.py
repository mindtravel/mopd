"""mopd.pool daemon end to end with real subprocesses on fake GPUs (no CUDA): dependencies, done
checks, failures blocking dependents, and a gang job preempting a backfilled job, which reruns."""

import importlib, json, os, sys, tempfile, time, unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


class DaemonTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["MOPD_ROOT"] = self.tmp.name
        os.environ["PYTHONPATH"] = str(SRC)
        import mopd.paths, mopd.pool.store, mopd.pool.daemon
        for m in (mopd.paths, mopd.pool.store, mopd.pool.daemon):
            importlib.reload(m)
        self.store, self.dm = mopd.pool.store, mopd.pool.daemon
        mopd.paths.TRAIN_PY = Path(sys.executable)
        self.dm.paths.TRAIN_PY = Path(sys.executable)
        from mopd.runtime.scheduler import GpuLease
        lease_dir = Path(self.tmp.name) / "leases"
        self.dm.GpuLease = lambda: GpuLease(lease_dir)
        self.dm.gpu_memory_used = lambda gpus=None: {g: 0 for g in (gpus or [0, 1])}
        self.dm.TICK = 0.05

    def tearDown(self):
        os.environ.pop("MOPD_ROOT", None)
        self.tmp.cleanup()

    def job(self, jid, code, **kw):
        return {"id": jid, "commands": [[[sys.executable, "-c", code], {}]], "log": f"{self.tmp.name}/logs/{jid.replace('/', '_')}.log", **kw}

    def run_until(self, d, cond, timeout=30):
        end = time.time() + timeout
        while time.time() < end:
            d.tick()
            if cond():
                return
            time.sleep(0.05)
        self.fail(f"timeout; states {[(j['id'], j['state']) for j in d.jobs.values()]}")

    def test_deps_done_checks_and_blocking(self):
        root = self.tmp.name
        self.store.submit([
            self.job("s/a", f"open('{root}/a','w').write('x')", gpus={"count": 1}, done={"kind": "file", "path": f"{root}/a"}),
            self.job("s/b", "import sys; sys.exit(3)", gpus=0, deps=["s/a"]),
            self.job("s/c", "pass", gpus=0, deps=["s/b"]),
            self.job("s/pre", "raise SystemExit('must not run')", gpus=0, done={"kind": "file", "path": f"{root}/pre"}),
        ])
        Path(root, "pre").write_text("already")
        d = self.dm.Daemon(gpus=[0, 1])
        self.run_until(d, lambda: all(j["state"] in ("done", "failed", "blocked") for j in d.jobs.values()) and len(d.jobs) == 4)
        st = {k: v["state"] for k, v in d.jobs.items()}
        self.assertEqual(st, {"s/a": "done", "s/b": "failed", "s/c": "blocked", "s/pre": "done"})
        self.assertTrue(d.lease.try_take([0, 1]))  # every lease given back when its job ended

    def test_submissions_in_one_second_all_arrive(self):
        for i in range(3):
            self.store.submit([self.job(f"s/{i}", "pass", gpus=0)])
        d = self.dm.Daemon(gpus=[0])
        d.ingest()
        self.assertEqual(sorted(d.jobs), ["s/0", "s/1", "s/2"])

    def test_gang_preempts_backfill_which_then_finishes(self):
        root = self.tmp.name
        long_eval = f"import time,os\nopen('{root}/eval-runs','a').write('r')\nfor _ in range(600): time.sleep(0.05)"
        self.store.submit([self.job("s/eval", long_eval, gpus={"min": 1, "max": 2}, priority=10, preemptible=True)])
        d = self.dm.Daemon(gpus=[0, 1])
        self.run_until(d, lambda: d.jobs["s/eval"]["state"] == "running")
        self.assertEqual(d.jobs["s/eval"]["assigned"], [0, 1])  # backfilled every free GPU
        self.store.submit([self.job("s/train", "pass", gpus={"count": 2}, priority=50)])
        self.run_until(d, lambda: d.jobs.get("s/train", {}).get("state") == "done")
        runs = Path(root, "eval-runs")
        # preempted once (possibly before it wrote anything), started again after the training
        self.run_until(d, lambda: d.jobs["s/eval"]["attempts"] == 2 and runs.exists())
        self.assertGreater(d.jobs["s/eval"]["started"], d.jobs["s/train"]["finished"] - 1)


if __name__ == "__main__":
    unittest.main()
