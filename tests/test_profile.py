"""mopd.profiling: FLOP / byte arithmetic and utilization against hand-computed values (CPU, no model)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import unittest
from types import SimpleNamespace

from mopd.profiling.model import ModelShape, infer_utilization, opd_utilization
from mopd.profiling.training import StepProfile

PEAKS = {"bf16_tflops": 200.0, "hbm_gbps": 1600}


class ProfileTest(unittest.TestCase):
    def shape(self):
        cfg = SimpleNamespace(num_hidden_layers=36, num_attention_heads=32, num_key_value_heads=8, head_dim=128, hidden_size=2560)
        return ModelShape(cfg, params=4_000_000_000)

    def test_train_flops_match_hand_computation(self):
        sh = self.shape()
        tokens, sum_sq = 4000, 1000**2 + 3000**2  # sequences of 1000 and 3000 tokens
        attention = 6 * 36 * (32 * 128) * sum_sq
        self.assertAlmostEqual(sh.train_flops(tokens, sum_sq), 6 * 4e9 * 4000 + attention, delta=1)
        self.assertAlmostEqual(sh.train_flops(tokens, sum_sq, recompute=True), 8 * 4e9 * 4000 + attention, delta=1)

    def test_config_dict_and_kv_width(self):
        sh = ModelShape({"num_hidden_layers": 36, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128}, 4e9)
        self.assertEqual(sh.kv_bytes_per_token, 2 * 36 * 8 * 128 * 2)

    def test_hybrid_multimodal_config(self):
        # Qwen3.5: text_config nested, 1 in 4 layers full attention
        cfg = {"text_config": {"num_hidden_layers": 32, "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256,
                               "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 8}}
        sh = ModelShape(cfg, 4e9)
        self.assertEqual(sh.layers, 8)
        self.assertEqual(sh.kv_bytes_per_token, 2 * 8 * 4 * 256 * 2)

    def test_infer_utilization(self):
        sh = self.shape()
        r = infer_utilization(sh, seconds=10, gpus=1, generated=5000, prefilled=1000, steps=500, context_tokens=10**7, peaks=PEAKS)
        flops = 2 * 4e9 * 6000 + 4 * 36 * 4096 * 10**7
        self.assertAlmostEqual(r["mfu"], flops / 10 / 1e12 / 200, places=4)
        moved = 500 * 8e9 + 10**7 * sh.kv_bytes_per_token
        self.assertAlmostEqual(r["mbu"], moved / 10 / 1e9 / 1600, places=4)
        self.assertEqual(r["tokens_per_step"], 12.0)
        # without the context only the weight traffic is known: reported as a lower bound
        self.assertIn("mbu_weights", infer_utilization(sh, seconds=10, gpus=1, generated=5000, steps=500, peaks=PEAKS))

    def test_opd_utilization_groups(self):
        r = opd_utilization(self.shape(), [1000, 3000], seconds={"step": 20, "student": 5, "generation": 12},
                            gpus={"student": 4, "rollout": 2}, rollout={"generated": 8000, "steps": 1000}, peaks=PEAKS)
        self.assertAlmostEqual(r["student_mfu_whole_step"] * 20, r["student_mfu_in_phase"] * 5, places=2)
        self.assertEqual(r["rollout_busy_fraction"], 0.6)
        self.assertEqual(r["rollout_generated_tok_s"], round(8000 / 12 / 2, 1))
        self.assertIn("rollout_mbu_weights", r)

    def test_step_profile_mfu(self):
        sh = self.shape()
        prof = StepProfile(sh, gpus=8, recompute=True, peak_tflops=200.0, device="cpu")
        prof.add_batch([1000, 3000], width=3000)
        prof.start -= 10.0  # pretend the step took 10 s
        r = prof.finish()
        self.assertAlmostEqual(r["padding_fraction"], 1 - 4000 / 6000, places=4)
        expected = sh.train_flops(4000, 1000**2 + 3000**2) / 10.0 / 8 / 1e12
        self.assertAlmostEqual(r["mfu"], expected / 200.0, places=3)
        self.assertGreater(r["hfu"], r["mfu"])


if __name__ == "__main__":
    unittest.main()
