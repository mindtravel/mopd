import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "scripts")]
import unittest
import torch
from transformers import AutoTokenizer
from mopd.training.dual_template import teacher_inputs, response_logps, policy_loss


class AlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = AutoTokenizer.from_pretrained("/ssd/lzx/mopd/models/Qwen3-4B")

    def test_answer_tokens_are_not_rewritten(self):
        for text in ["  leading spaces\n", "<think>raw</think>\\boxed{7}", " Question: same words"]:
            answer = torch.tensor([self.tok.encode(text, add_special_tokens=False)])
            ids, start = teacher_inputs(self.tok, "Question: same words", answer, "cpu")
            self.assertTrue(torch.equal(ids[:, start:], answer))
            self.assertIn("</think>", self.tok.decode(ids[0, :start]))

    def test_causal_shift_and_teacher_detach(self):
        ids = torch.tensor([[0, 1, 2, 3]])
        s = torch.randn(1, 4, 5, requires_grad=True)
        t = torch.randn(2, requires_grad=True)
        lp = response_logps(s, ids, 2)
        expected = s[0, 1:3].log_softmax(-1).gather(1, ids[0, 2:, None]).squeeze(1)
        torch.testing.assert_close(lp, expected)
        loss, stats = policy_loss(lp, t)
        loss.backward()
        self.assertIsNone(t.grad)
        self.assertGreater(s.grad.abs().sum().item(), 0)
        self.assertEqual(s.grad[0, 0].abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
