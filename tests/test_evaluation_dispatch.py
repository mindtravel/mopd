import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "scripts")]
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_model_version as dispatcher


class EvaluationDispatchTests(unittest.TestCase):
    def write_manifest(self, root, student, checkpoints=None):
        run = root / "runs/fairy-run"
        run.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema": "ifairy.model_version.v1",
            "run_id": "fairy-run",
            "version_id": "fairy-run",
            "components": {"student": student},
            "checkpoints": checkpoints or [],
        }
        path = run / "model-version.json"
        path.write_text(json.dumps(manifest))
        data = root / "data/test.jsonl"
        data.parent.mkdir(parents=True)
        data.write_text("{}\n")
        return run, path, data

    def args(self, manifest, data, checkpoint="base", backend="auto", out=None, student_prompt_format="plain_qa_v1"):
        return SimpleNamespace(
            model_version=str(manifest),
            checkpoint=checkpoint,
            data=str(data),
            protocol_id="ifeval-v1",
            out=out,
            backend=backend,
            device=[3],
            max_new_tokens=128,
            max_model_len=4096,
            max_seqs=4,
            batch_size=2,
            max_num_batched_tokens=2048,
            seed=7,
            student_prompt_format=student_prompt_format,
        )

    def test_auto_dispatch_uses_native_hf_for_fairy2i(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, manifest, data = self.write_manifest(
                root,
                {
                    "base_model": "/models/fairy2i",
                    "backend": "fairy2i_qat",
                },
            )
            with patch.object(dispatcher, "ROOT", root):
                command, env, output, backend = dispatcher.build_command(self.args(manifest, data))

        self.assertEqual(backend, "hf_qat")
        self.assertIn("generate_benchmark_hf_qat.py", command[1])
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "3")
        self.assertTrue(output.is_relative_to(run))

    def test_auto_dispatch_keeps_lora_on_vllm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "runs/fairy-run"
            checkpoint_path = run / "checkpoint-0001"
            checkpoint_path.mkdir(parents=True)
            student = {"base_model": "/models/qwen", "backend": "hf_lora"}
            checkpoints = [
                {
                    "version_id": "fairy-run:step-0001",
                    "path": str(checkpoint_path),
                    "artifact_type": "lora_adapter",
                    "complete": True,
                }
            ]
            run, manifest, data = self.write_manifest(root, student, checkpoints)
            with patch.object(dispatcher, "ROOT", root):
                command, _, _, backend = dispatcher.build_command(
                    self.args(manifest, data, checkpoint="checkpoint-0001")
                )

        self.assertEqual(backend, "vllm")
        self.assertIn("generate_benchmark_vllm.py", command[1])
        self.assertIn(str(checkpoint_path), command)

    def test_rejects_backend_that_cannot_load_qat_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, manifest, data = self.write_manifest(
                root,
                {
                    "base_model": "/models/fairy2i",
                    "backend": "fairy2i_qat",
                },
                [
                    {
                        "version_id": "fairy-run:step-0001",
                        "path": str(root / "update"),
                        "artifact_type": "fairy2i_qat_scale_update",
                        "complete": True,
                    }
                ],
            )
            with patch.object(dispatcher, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "vLLM cannot load"):
                    dispatcher.build_command(
                        self.args(manifest, data, checkpoint="fairy-run:step-0001", backend="vllm")
                    )

    def test_forwards_recorded_student_prompt_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, manifest, data = self.write_manifest(
                root,
                {
                    "base_model": "/models/qwen",
                    "backend": "hf_lora",
                    "prompt_format": "plain_qa_v1",
                },
            )
            with patch.object(dispatcher, "ROOT", root):
                command, _, _, _ = dispatcher.build_command(self.args(manifest, data, student_prompt_format=None))

        index = command.index("--student-prompt-format")
        self.assertEqual(command[index + 1], "plain_qa_v1")

    def test_rejects_prompt_format_that_differs_from_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, manifest, data = self.write_manifest(
                root,
                {
                    "base_model": "/models/qwen",
                    "backend": "hf_lora",
                    "prompt_format": "plain_qa_v1",
                },
            )
            with patch.object(dispatcher, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "trained with student prompt"):
                    dispatcher.build_command(self.args(manifest, data, student_prompt_format="chat_template_v1"))

    def test_legacy_manifest_requires_explicit_prompt_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, manifest, data = self.write_manifest(
                root,
                {
                    "base_model": "/models/qwen",
                    "backend": "hf_lora",
                },
            )
            with patch.object(dispatcher, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "no student prompt_format"):
                    dispatcher.build_command(self.args(manifest, data, student_prompt_format=None))


if __name__ == "__main__":
    unittest.main()
