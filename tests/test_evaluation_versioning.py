import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "scripts")]
import tempfile
import unittest
from pathlib import Path

from mopd.evaluation.versioning import resolve_checkpoint


class ResolveCheckpointTests(unittest.TestCase):
    def test_resolves_base_as_distinct_model_version(self):
        manifest = {
            "run_id": "fairy-run",
            "components": {"student": {"base_model": "/models/fairy"}},
        }

        resolved = resolve_checkpoint(manifest, "base")

        self.assertEqual(resolved["version_id"], "fairy-run:base")
        self.assertEqual(resolved["base_model"], "/models/fairy")
        self.assertIsNone(resolved["path"])

    def test_resolves_complete_checkpoint_by_directory_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint-0002"
            checkpoint.mkdir()
            manifest = {
                "run_id": "fairy-run",
                "components": {"student": {"base_model": "/models/fairy"}},
                "checkpoints": [{
                    "version_id": "fairy-run:step-0002",
                    "path": str(checkpoint),
                    "artifact_type": "fairy2i_qat_scale_update",
                    "complete": True,
                }],
            }

            resolved = resolve_checkpoint(manifest, "checkpoint-0002")

        self.assertEqual(resolved["version_id"], "fairy-run:step-0002")
        self.assertEqual(resolved["artifact_type"], "fairy2i_qat_scale_update")

    def test_rejects_incomplete_checkpoint(self):
        manifest = {
            "run_id": "fairy-run",
            "components": {"student": {"base_model": "/models/fairy"}},
            "checkpoints": [{
                "version_id": "fairy-run:step-0002",
                "path": "/runs/fairy-run/checkpoint-0002",
                "complete": False,
            }],
        }

        with self.assertRaisesRegex(ValueError, "incomplete"):
            resolve_checkpoint(manifest, "fairy-run:step-0002")


if __name__ == "__main__":
    unittest.main()
