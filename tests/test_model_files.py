"""model_files.finalize on tiny synthetic directories (needs torch + safetensors)."""
import json, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from safetensors.torch import save_file
from mopd.runtime.model_files import finalize


def write_dir(d, tensors, config, extra_files=()):
    d.mkdir(parents=True)
    save_file(tensors, str(d / "model.safetensors"))
    (d / "config.json").write_text(json.dumps(config))
    for name, text in extra_files:
        (d / name).write_text(text)


class FinalizeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.embed = torch.randn(8, 4)
        write_dir(self.tmp / "base", {"model.embed_tokens.weight": self.embed, "model.norm.weight": torch.ones(4)},
                  {"tie_word_embeddings": True, "rope_theta": 1000000}, [("tokenizer.json", "{}")])

    def test_drops_identical_tied_lm_head_and_copies_base_files(self):
        out = self.tmp / "out"
        write_dir(out, {"model.embed_tokens.weight": self.embed, "model.norm.weight": torch.ones(4), "lm_head.weight": self.embed.clone()},
                  {"tie_word_embeddings": True, "rope_parameters": {"rope_theta": 1000000}})
        r = finalize(out, self.tmp / "base")
        self.assertEqual(r["dropped"], "lm_head.weight"); self.assertEqual(r["tensors"], 2)
        self.assertEqual(json.loads((out / "config.json").read_text()), {"tie_word_embeddings": True, "rope_theta": 1000000})
        self.assertTrue((out / "tokenizer.json").exists())
        finalize(out, self.tmp / "base")  # idempotent

    def test_rejects_shape_or_name_mismatch(self):
        out = self.tmp / "bad"
        write_dir(out, {"model.embed_tokens.weight": torch.randn(9, 4), "model.norm.weight": torch.ones(4)}, {})
        with self.assertRaises(ValueError):
            finalize(out, self.tmp / "base")
        out2 = self.tmp / "bad2"
        write_dir(out2, {"model.embed_tokens.weight": self.embed, "model.norm.weight": torch.ones(4), "lm_head.weight": torch.zeros(8, 4)},
                  {"tie_word_embeddings": True})
        with self.assertRaises(ValueError):  # a tied head that is not the embedding is not silently dropped
            finalize(out2, self.tmp / "base")


if __name__ == "__main__":
    unittest.main()
