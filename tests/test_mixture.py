import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "scripts")]
import json
import tempfile
import unittest
from pathlib import Path

from mopd.data.mixture import plan_domains, quotas, resolve_domain


def write_pool(path, prefix, n):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"id": f"{prefix}-{i}", "domain": "instruction", "split": "train",
                                        "prompt": f"{prefix} prompt number {i}"}) + "\n" for i in range(n)))


class MixtureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write_pool(self.root / "data/a.jsonl", "a", 40)
        write_pool(self.root / "data/b.jsonl", "b", 40)
        (self.root / "configs/mixtures").mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def preset(self, name, sources, domain="instruction"):
        (self.root / f"configs/mixtures/{name}.json").write_text(json.dumps({"domain": domain, "sources": sources}))

    def config(self, spec, steps=4):
        return {"seed": 7, "steps": steps, "domains": {"instruction": resolve_domain("instruction", spec, self.root)}}

    def test_quotas_split_by_weight_with_largest_remainder(self):
        sources = [{"name": "a", "weight": 2}, {"name": "b", "weight": 1}]
        self.assertEqual(quotas(16, sources), {"a": 11, "b": 5})
        self.assertEqual(quotas(3, sources), {"a": 2, "b": 1})
        with self.assertRaises(ValueError):
            quotas(1, sources)

    def test_every_step_has_exact_per_source_quota(self):
        self.preset("ab", [{"name": "a", "path": "data/a.jsonl", "weight": 1},
                           {"name": "b", "path": "data/b.jsonl", "weight": 3}])
        plan = plan_domains(self.config({"mixture": "ab", "count": 8}))
        for batch in plan:
            sources = [r["mixture_source"] for r in batch]
            self.assertEqual((sources.count("a"), sources.count("b")), (2, 6))
        ids = [r["id"] for batch in plan for r in batch]
        self.assertEqual(len(ids), len(set(ids)))

    def test_single_source_preset_matches_legacy_train_file(self):
        self.preset("a", [{"name": "a", "path": "data/a.jsonl", "weight": 1}])
        legacy = plan_domains(self.config({"train_file": str(self.root / "data/a.jsonl"), "count": 8}))
        preset = plan_domains(self.config({"mixture": "a", "count": 8}))
        self.assertEqual([[r["id"] for r in b] for b in legacy], [[r["id"] for r in b] for b in preset])

    def test_rejects_overlapping_sources_and_wrong_domain(self):
        self.preset("aa", [{"name": "a1", "path": "data/a.jsonl", "weight": 1},
                           {"name": "a2", "path": "data/a.jsonl", "weight": 1}])
        with self.assertRaises(ValueError):
            plan_domains(self.config({"mixture": "aa", "count": 4}))
        self.preset("m", [{"name": "a", "path": "data/a.jsonl", "weight": 1}], domain="math")
        with self.assertRaises(ValueError):
            resolve_domain("instruction", {"mixture": "m", "count": 4}, self.root)
        with self.assertRaises(ValueError):
            resolve_domain("instruction", {"mixture": "m", "train_file": "x", "count": 4}, self.root)

    def test_pool_exhaustion_is_per_source(self):
        self.preset("ab", [{"name": "a", "path": "data/a.jsonl", "weight": 1},
                           {"name": "b", "path": "data/b.jsonl", "weight": 1}])
        with self.assertRaises(ValueError):
            plan_domains(self.config({"mixture": "ab", "count": 8}, steps=11))


if __name__ == "__main__":
    unittest.main()
