import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "scripts")]
import sys
import unittest
from pathlib import Path

from mopd.evaluation.generation_manifest import build_generation_manifest


class GenerationManifestTests(unittest.TestCase):
    def build(self, *, model_version_id="run:base", data_sha256="data-a"):
        return build_generation_manifest(
            run_id="run", model_version_id=model_version_id,
            protocol_id="ifeval-v1", prompt_mode="student-prompt-v1",
            generation={"temperature": 0, "max_new_tokens": 128, "seed": 7},
            data_path="/project/data.jsonl", data_sha256=data_sha256,
            backend="hf_fairy2i_qat", weight_representation="qat-base",
            inference_dtype="bf16-activation-1bit-linear",
            model={"base_model": "/models/student", "checkpoint_path": None},
        )

    def test_protocol_hash_is_separate_from_model_and_dataset(self):
        first = self.build()
        second = self.build(model_version_id="run:step-0001", data_sha256="data-b")

        self.assertEqual(first["protocol_sha256"], second["protocol_sha256"])
        self.assertNotEqual(first["model_version_id"], second["model_version_id"])
        self.assertNotEqual(first["dataset"]["sha256"], second["dataset"]["sha256"])

    def test_required_attribution_fields_are_backend_neutral(self):
        manifest = self.build()

        self.assertEqual(manifest["schema"], "ifairy.benchmark_generation.v1")
        self.assertEqual(manifest["protocol"], {
            "protocol_id": "ifeval-v1",
            "prompt_mode": "student-prompt-v1",
            "generation": {"temperature": 0, "max_new_tokens": 128, "seed": 7},
        })
        self.assertEqual(manifest["dataset"]["path"], "/project/data.jsonl")


if __name__ == "__main__":
    unittest.main()
