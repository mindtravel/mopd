import unittest

from mopd.data.domains import student_prompt


class FakeTokenizer:
    def __init__(self):
        self.call = None

    def apply_chat_template(self, messages, **kwargs):
        self.call = (messages, kwargs)
        return "<|im_start|>user\nOnly output OK<|im_end|>\n<|im_start|>assistant\n"


class StudentPromptTests(unittest.TestCase):
    def test_chat_template_format_uses_checkpoint_chat_template(self):
        tokenizer = FakeTokenizer()
        row = {"domain": "instruction", "prompt": "Only output OK"}

        rendered = student_prompt(tokenizer, row, "chat_template_v1")

        self.assertEqual(
            rendered,
            "<|im_start|>user\nOnly output OK<|im_end|>\n<|im_start|>assistant\n",
        )
        self.assertEqual(
            tokenizer.call,
            (
                [{"role": "user", "content": "Only output OK"}],
                {
                    "tokenize": False,
                    "add_generation_prompt": True,
                    "enable_thinking": False,
                },
            ),
        )

    def test_plain_qa_format_matches_dual_tulu3_run(self):
        tokenizer = FakeTokenizer()

        instruction = student_prompt(tokenizer, {"domain": "instruction", "prompt": "Only output OK"}, "plain_qa_v1")
        math = student_prompt(tokenizer, {"domain": "math", "prompt": "1+1?"}, "plain_qa_v1")

        self.assertEqual(instruction, "Instruction: Only output OK\nAnswer:")
        self.assertEqual(math, "Question: 1+1?\nSolve step by step and put your final answer in \\boxed{}.\nAnswer:")
        self.assertIsNone(tokenizer.call)

    def test_format_is_required(self):
        row = {"domain": "instruction", "prompt": "x"}
        with self.assertRaises(TypeError):
            student_prompt(FakeTokenizer(), row)
        with self.assertRaises(ValueError):
            student_prompt(FakeTokenizer(), row, "chat")


if __name__ == "__main__":
    unittest.main()
