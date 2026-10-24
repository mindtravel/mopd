"""mopd.runtime.profile: FLOP arithmetic and MFU against hand-computed values (CPU, no model)."""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "scripts")]
import sys, unittest
from pathlib import Path
from types import SimpleNamespace
from mopd.runtime.profile import ModelShape, StepProfile


class ProfileTest(unittest.TestCase):
    def shape(self):
        cfg = SimpleNamespace(num_hidden_layers=36, num_attention_heads=32, head_dim=128, hidden_size=2560)
        return ModelShape(cfg, params=4_000_000_000)

    def test_flops_match_hand_computation(self):
        sh = self.shape()
        # two sequences of 1000 and 3000 tokens
        tokens, sum_sq = 4000, 1000 ** 2 + 3000 ** 2
        dense = 6 * 4e9 * 4000
        attention = 6 * 36 * (32 * 128) * sum_sq
        self.assertAlmostEqual(sh.flops(tokens, sum_sq), dense + attention, delta=1)
        self.assertAlmostEqual(sh.flops(tokens, sum_sq, recompute=True), 8 * 4e9 * 4000 + attention, delta=1)

    def test_mfu_uses_measured_peak_and_all_gpus(self):
        sh = self.shape()
        prof = StepProfile(sh, gpus=8, recompute=True, peak_tflops=200.0, device="cpu")
        prof.add_batch([1000, 3000], width=3000)
        prof.start -= 10.0  # pretend the step took 10 s
        r = prof.finish()
        self.assertEqual(r["tokens"], 4000)
        self.assertAlmostEqual(r["padding_fraction"], 1 - 4000 / 6000, places=4)
        expected = sh.flops(4000, 1000 ** 2 + 3000 ** 2) / 10.0 / 8 / 1e12
        self.assertAlmostEqual(r["model_tflops_per_gpu"], expected, places=1)
        self.assertAlmostEqual(r["mfu"], expected / 200.0, places=3)
        self.assertGreater(r["hfu"], r["mfu"])

    def test_no_peak_means_no_mfu(self):
        prof = StepProfile(self.shape(), gpus=1, recompute=False, peak_tflops=None, device="cpu")
        prof.add_batch([10], width=10)
        r = prof.finish()
        self.assertNotIn("mfu", r); self.assertIn("model_tflops_per_gpu", r)


if __name__ == "__main__":
    unittest.main()
