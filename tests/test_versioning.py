import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "scripts")]
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from mopd.evaluation.versioning import write_manifest


class ModelVersionInputTests(unittest.TestCase):
    def test_training_inputs_are_fingerprinted_once_per_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "runs" / "trial"
            run.mkdir(parents=True)
            train = root / "data" / "math.jsonl"
            holdout = root / "data" / "math-dev.jsonl"
            train.parent.mkdir()
            train.write_text('{"prompt":"solve"}\n')
            holdout.write_text('{"prompt":"validate"}\n')
            config = {
                "student": "student-base",
                "steps": 2,
                "domains": {
                    "math": {
                        "train_file": str(train),
                        "holdout_files": [str(holdout)],
                    }
                },
            }
            config_path = run / "config.json"
            config_path.write_text(json.dumps(config, sort_keys=True))

            first_path = write_manifest(config, run, {"state": "training", "step": 1})
            first = json.loads(first_path.read_text())
            inputs = first["training"]["inputs"]
            train_record = inputs["domains"]["math"]["train"]
            self.assertEqual(train_record["sha256"], hashlib.sha256(train.read_bytes()).hexdigest())
            self.assertEqual(train_record["size_bytes"], train.stat().st_size)
            self.assertEqual(inputs["domains"]["math"]["holdouts"][0]["path"], str(holdout))
            self.assertEqual(
                first["training"]["config_sha256"],
                hashlib.sha256(config_path.read_bytes()).hexdigest(),
            )

            train.write_text('{"prompt":"changed after run start"}\n')
            second = json.loads(write_manifest(config, run, {"state": "training", "step": 2}).read_text())
            self.assertEqual(second["training"]["inputs"], inputs)

            config_path.write_text(json.dumps({**config, "seed": 99}, sort_keys=True))
            with self.assertRaisesRegex(ValueError, "Run config changed"):
                write_manifest(config, run, {"state": "training", "step": 3})


if __name__ == "__main__":
    unittest.main()
