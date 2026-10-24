"""Checkpoint benchmark service on para5090 (IFBench + AutomationBench); see service.py and
docs/evaluation/bench-service.md.

  protocol     fixed evaluation settings and request validation
  jobs         job records (runs/bench-service/jobs), re-queued after a restart
  engine       resident vLLM OpenAI server per service GPU: weight swap / LoRA / restart, counters
  weight_swap  vLLM worker extension that loads a checkpoint in place
  benchmarks   the two benchmarks in the mopd-bench docker image, result summaries
  service      workers on GPUs 6/7, HTTP API on 127.0.0.1:18090
"""
