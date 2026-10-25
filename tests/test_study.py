"""mopd.study without GPUs: grid expansion, restricted conditions, and identities that match the
older experiment configs (so finished runs are reused instead of retrained)."""

import json, sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mopd.study import resolve, spec

STUDY = ROOT / "configs/studies/full-sft-lr.yaml"


class StudyTest(unittest.TestCase):
    def test_grid_and_conditional_evals(self):
        st = spec.load(STUDY)
        runs = spec.expand(st)
        self.assertEqual(len(runs), 15)
        self.assertEqual(spec.evals_for(st, dict(runs[-1][1])), ["unified-v3", "lcbdev-v3"])  # code
        self.assertEqual(spec.evals_for(st, dict(runs[0][1])), ["unified-v3"])  # math

    def test_conditions_are_restricted(self):
        self.assertTrue(spec.condition("domain in ['code', 'math'] and lr < 1e-5", {"domain": "code", "lr": 2e-6}))
        for bad in ("__import__('os')", "domain.upper()", "nope == 1"):
            with self.assertRaises(ValueError):
                spec.condition(bad, {"domain": "code"})

    def test_identity_matches_experiment_configs(self):
        st, reg = spec.load(STUDY), resolve.registry()
        old = {}
        for p in (ROOT / "configs/experiments").glob("*/*.json"):
            c = json.loads(p.read_text())
            if c.get("training_method") == "sft" and len(c.get("domains", {})) == 1:
                old[resolve.identity(c)] = p.parent.name + "/" + p.stem
        for name, fields in spec.expand(st):
            cfg = resolve.run_config(st, name, fields, spec.evals_for(st, fields), "s", "h", reg)
            self.assertIn(resolve.identity(cfg), old, name)
        # math at 1e-5 is full256's expert (same data, seed, controls) -- reused, not retrained
        cfg = resolve.run_config(st, "m", {"method": "sft", "domain": "math", "lr": 1e-5}, ["unified-v3"], "s", "h", reg)
        self.assertEqual(old[resolve.identity(cfg)], "full256/expert-m")

    def test_identity_ignores_system_fields(self):
        st, reg = spec.load(STUDY), resolve.registry()
        f = {"method": "sft", "domain": "code", "lr": 2e-6}
        a = resolve.run_config(st, "a", f, [], "s", "h", reg)
        b = resolve.run_config(st, "b", {**f, "micro_batch_size": 8, "fsdp": {"offload": True}}, [], "s", "h", reg)
        self.assertEqual(resolve.identity(a), resolve.identity(b))
        c = resolve.run_config(st, "c", {**f, "seed": 1}, [], "s", "h", reg)
        self.assertNotEqual(resolve.identity(a), resolve.identity(c))


    def test_mopd_layout_and_identity(self):
        st = spec.load(ROOT / "configs/studies/mopd-mic.yaml")
        reg = resolve.registry()
        (name, fields), = spec.expand(st)
        teachers = {d: f"/runs/{d}/checkpoint-0256" for d in ("math", "instruction", "code")}
        cfg = resolve.run_config(st, name, fields, ["unified-v3"], "s", "h", reg, teachers)
        self.assertEqual((cfg["student_gpus"], cfg["rollout_devices"]), (2, ["cuda:2", "cuda:3"]))
        self.assertEqual({s["device"] for s in cfg["domains"].values()}, {"cuda:0", "cuda:1"})  # on the student GPUs
        self.assertEqual([s["count"] for s in cfg["domains"].values()], [11, 11, 10])
        self.assertTrue(cfg["rollout_async"] and cfg["rollout_streaming"] and cfg["fsdp"]["offload"])
        other = resolve.run_config(st, name, fields, [], "s", "h", reg, {**teachers, "code": "/runs/other/checkpoint-0256"})
        self.assertNotEqual(resolve.identity(cfg), resolve.identity(other))  # a different teacher is a different run
        faster = resolve.run_config(st, name, {**fields, "micro_batch_tokens": 4096}, [], "s", "h", reg, teachers)
        self.assertEqual(resolve.identity(cfg), resolve.identity(faster))


if __name__ == "__main__":
    unittest.main()
