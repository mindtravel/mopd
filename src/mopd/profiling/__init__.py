"""How much of the GPUs training and inference use, one component for the trainers, the MOPD rollout
engines and the bench service. Inference figures come from counters read once per step or job (the
engines count their own steps; the bench service diffs vLLM's /metrics), never from sampling, so
measuring costs nothing.

  peaks     measured bf16 TFLOPS / copy GB/s per GPU model (cache/profile)
  model     ModelShape, infer_utilization, opd_utilization            (no torch)
  training  StepProfile, torch_profile, trace_summary                  (torch)
"""
