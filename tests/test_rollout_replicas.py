import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "scripts")]
import unittest, torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from peft import LoraConfig, get_peft_model
from mopd.rollout.replicas import sync_lora, assign_shards


class Tests(unittest.TestCase):
    def test_sync_only_lora(self):
        def model():
            return get_peft_model(
                Qwen3ForCausalLM(
                    Qwen3Config(
                        vocab_size=32,
                        hidden_size=16,
                        intermediate_size=32,
                        num_hidden_layers=1,
                        num_attention_heads=2,
                        num_key_value_heads=1,
                        head_dim=8,
                    )
                ),
                LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
            )

        a, b = model(), model()
        frozen = {n: p.clone() for n, p in b.named_parameters() if not p.requires_grad}
        with torch.no_grad():
            for p in a.parameters():
                if p.requires_grad:
                    p.add_(0.25)
        sync_lora(a, b)
        for n, p in b.named_parameters():
            torch.testing.assert_close(p, dict(a.named_parameters())[n] if p.requires_grad else frozen[n])

    def test_assign_preserves_each_item_once(self):
        rows = list(range(8))
        shards = assign_shards(rows, 2)
        self.assertEqual(sorted(i for s in shards for i in s), rows)
        self.assertEqual(list(map(len, shards)), [4, 4])


if __name__ == "__main__":
    unittest.main()
