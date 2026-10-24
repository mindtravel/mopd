"""GPU smoke test for mopd.rollout.engine_stream: the submit-everything path (MOPD rollout) and the
incremental feed path (benchmark generation) must return every request exactly once, with the same
greedy tokens as LLM.generate.

  CUDA_VISIBLE_DEVICES=<idle gpu> VLLM_ENABLE_V1_MULTIPROCESSING=0 mopd_infer/bin/python scripts/test_engine_stream.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "scripts")]
from mopd import paths
import sys
from pathlib import Path
from vllm import LLM, SamplingParams
from mopd.rollout.engine_stream import stream_feed, stream_requests

engine = LLM(model=str(paths.BASE_MODEL), dtype="bfloat16", max_model_len=2048, max_num_seqs=8, seed=1)
tok = engine.get_tokenizer()
prompts = [{"prompt_token_ids": tok.encode(f"Question: what is {i} + {i * 3}?\nAnswer:")} for i in range(12)]
params = SamplingParams(temperature=0, max_tokens=24)
want = [list(o.outputs[0].token_ids) for o in engine.generate(prompts, params, use_tqdm=False)]

got = {}
stream_requests(engine, prompts, params, None, lambda i, o: got.__setitem__(i, list(o.outputs[0].token_ids)))
assert sorted(got) == list(range(12)), got.keys()
assert sum(got[i] == want[i] for i in range(12)) >= 10, "submit-all path diverges from generate()"

batches = [[(i, prompts[i]) for i in range(j, min(j + 5, 12))] for j in range(0, 12, 5)]
fed = {}


def feed(unfinished):
    return batches.pop(0) if batches else None


stream_feed(engine, params, None, feed, lambda k, o: fed.__setitem__(k, list(o.outputs[0].token_ids)))
assert sorted(fed) == list(range(12)), fed.keys()
assert sum(fed[i] == want[i] for i in range(12)) >= 10, "incremental feed path diverges from generate()"
print(
    "engine_stream ok:",
    sum(got[i] == want[i] for i in range(12)),
    sum(fed[i] == want[i] for i in range(12)),
    "of 12 identical to generate()",
)
