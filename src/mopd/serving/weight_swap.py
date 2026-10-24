"""vLLM worker extension for the bench service (scripts/bench_service.py): load another checkpoint's
weights into a running OpenAI server instead of restarting it (~1-2 min of startup and compilation
per checkpoint). Called through the server's dev-mode /collective_rpc endpoint:

  POST /collective_rpc {"method": "load_checkpoint", "args": ["/path/to/checkpoint"]}

Only valid for a checkpoint with the running model's architecture and tokenizer/chat template; the
service checks that (serving_signature) before choosing a swap over a restart.
"""

import glob
import os


class WeightSwap:
    def load_checkpoint(self, path):
        from vllm.model_executor.model_loader.weight_utils import safetensors_weights_iterator

        files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"no safetensors in {path}")
        loaded = self.model_runner.model.load_weights(safetensors_weights_iterator(files, False))
        return len(loaded) if loaded is not None else None
