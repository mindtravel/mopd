import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "scripts")]
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file

from mopd.backends.fairy2i_qat_adapter import (
    load_fairy2i_qat_update,
    save_fairy2i_qat_update,
)


class TinyQatModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Module()
        self.projection.weight = torch.nn.Parameter(torch.ones(3, 3))
        self.projection.quant_scale = torch.nn.Parameter(torch.ones(4, 1, 1))
        self.requires_grad_(False)
        self.projection.quant_scale.requires_grad_(True)


class QatUpdateTests(unittest.TestCase):
    def test_round_trip_restores_only_quant_scales(self):
        model = TinyQatModel()
        original_scale = model.projection.quant_scale.detach().clone()
        with tempfile.TemporaryDirectory() as temp:
            save_fairy2i_qat_update(model, temp, base_model="fairy2i")
            with torch.no_grad():
                model.projection.weight.fill_(7)
                model.projection.quant_scale.zero_()
            current_weight = model.projection.weight.detach().clone()

            load_fairy2i_qat_update(model, temp)
            saved = load_file(str(Path(temp) / "qat_scales.safetensors"))

        self.assertTrue(torch.equal(model.projection.weight, current_weight))
        self.assertTrue(torch.equal(model.projection.quant_scale, original_scale))
        self.assertEqual(set(saved), {"projection.quant_scale"})

    def test_restore_rejects_a_different_scale_layout(self):
        with tempfile.TemporaryDirectory() as temp:
            save_fairy2i_qat_update(TinyQatModel(), temp, base_model="fairy2i")
            target = TinyQatModel()
            target.projection.quant_scale = torch.nn.Parameter(torch.ones(4, 2, 1))

            with self.assertRaisesRegex(ValueError, "quant_scale"):
                load_fairy2i_qat_update(target, Path(temp))

    def test_save_rejects_other_trainable_parameters(self):
        model = TinyQatModel()
        model.projection.weight.requires_grad_(True)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "Only quant_scale"):
                save_fairy2i_qat_update(model, temp, base_model="fairy2i")


if __name__ == "__main__":
    unittest.main()
