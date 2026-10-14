# MOPD Benchmark and Model-Version Contract

Benchmark generation, scoring, and dashboard display are separate from MOPD training. A result is comparable only when its model version, benchmark protocol, data digest, and inference backend are recorded.

## Model versions

Each training run writes `model-version.json` with schema `ifairy.model_version.v1`. The run ID groups the recipe; each complete checkpoint gets an ID such as `<run-id>:step-0002`. The manifest records the student base model, update type, forward precision, quantization settings, frozen teachers, rollout backend, checkpoint path, and completeness.

`scripts/evaluate_model_version.py` resolves either `base` or a complete checkpoint from that manifest. Its `auto` route selects the native HF QAT implementation for Fairy2i and vLLM for compatible base or LoRA weights. Incompatible representations fail before model loading; a QAT scale update is not presented to vLLM as a LoRA adapter.

## Benchmark records

Every new `generation-config.json` uses schema `ifairy.benchmark_generation.v1` and the same attribution fields for HF-QAT and vLLM:

- `model_version_id` identifies the exact base or checkpoint.
- `protocol_id` and `protocol_sha256` identify prompt formatting and generation settings. The protocol digest is independent of the model and dataset.
- `dataset.path` and `dataset.sha256` identify the exact input file.
- `backend`, `weight_representation`, and `inference_dtype` describe the runtime actually used.
- `model` records the base and optional checkpoint artifact, including its digest.

Generation writes resumable `generated.jsonl` records and atomic progress files. `scripts/grade_benchmark_outputs.py` scores those outputs independently and repeats model-version and protocol IDs on every scored case. The monitor is read-only and never starts inference or grading.

New evaluations are stored under the training run at `evaluations/<checkpoint>/<protocol-id>/`. The dashboard discovers training runs from their manifests and displays only benchmark output whose `model_version_id` belongs to the selected run. Historical benchmark folders without a version ID remain intact but are not attributed to a newer run.

## Backend boundaries

MOPD owns the training forward, teacher scoring, and differentiable student log probabilities. Evaluation adapters own generation only; all adapters feed the same normalized records and scorer.

- Implemented: native HF generation for Fairy2i QAT checkpoints, including scale-only update restore; vLLM generation for compatible base/LoRA checkpoints; shared scorer; dynamic per-run monitor and version-scoped benchmark listing.
- Read-only references: `/root/lzx/packed_work/4b_eval/packed_adapter/benchmark_queue.py`, `current_batch_worker.py`, and `ablate_protocol/benchmark_runtime/` contain legacy scheduling, HF, and packed-server patterns.
- Not yet connected: the packed-server worker and a GPU-aware batch queue that schedules multiple model/protocol pairs. Legacy hard-coded GPU allowlists and fixed benchmark sets are not copied into MOPD.

The target experiment is the Fairy2i quantized student, not merged weights. Its checkpoints freeze the BF16 source tensors and update only the 252 `quant_scale` tensors; an HF QAT checkpoint, packed inference artifact, and BF16 LoRA checkpoint are distinct representations and must be evaluated by compatible backends under the same protocol before comparison.

## Fairy2i validation state

The training environment pins Transformers 5.2.0 for the custom Qwen3 code shipped with the Fairy2i checkpoint. `mopd_infer` remains separate for vLLM. A MOPD-owned backward fallback replaces the unsupported Triton scale-gradient API without modifying the read-only source kernel.

The two-step mixed Numina + Tülu 3 smoke completed: both teachers stayed frozen, scale-update L2 was `0.000140679`, and checkpoints contain only scale updates. It processed four prompts per domain, capped generations at 64 tokens, and all eight outputs hit that cap; this proves the training path connects, not that the model improved.

The matched 32-question IFEval run for the base and step-2 checkpoint is currently testing the new generation, scoring, and dashboard attribution path. The result is a smoke benchmark, not a full IFEval claim.
