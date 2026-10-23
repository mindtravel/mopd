"""mopd.evaluation.context.default_max_model_len: the single source of truth for vLLM's context
length, so no caller can hardcode a stale value that silently truncates once max_new_tokens grows
past it (what generate_benchmark_vllm.py did before this existed)."""
import unittest
from mopd.evaluation.context import default_max_model_len


class DefaultMaxModelLenTest(unittest.TestCase):
    def test_floor_for_existing_protocols(self):
        # every protocol already in use (4096 and 8192 max_new_tokens) must keep today's 16384
        self.assertEqual(default_max_model_len(4096), 16384)
        self.assertEqual(default_max_model_len(8192), 16384)

    def test_grows_with_generation_budget(self):
        self.assertEqual(default_max_model_len(24576), 24576 + 4096)
        self.assertGreater(default_max_model_len(20000), 20000)  # always room for a nonempty prompt


if __name__ == "__main__":
    unittest.main()
