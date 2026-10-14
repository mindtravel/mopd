# NuminaMath OPD experiment

Baseline: student 68/128; teacher 102/128. Fixed MATH test subset, not full benchmark.

Training config: configs/numina_opd.json. Real NuminaMath olympiads/amc_aime only; data provenance and duplicate screen: data/numina/download.json and selection.json. No reference solutions in training input. Semantic contamination is not ruled out.

64 updates x 4 prompts, fresh Base LoRA, max 1024 new tokens, GPU4/5. First step verified: 2185 tokens, no truncation, finite nonzero gradient. Training remains in progress.

Outputs: runs/20261013-numina-competition-opd. Final adapter checkpoint-0064 automatically evaluated via scripts/run_numina_experiment.py. Comparison: runs/20261013-math-evaluation/numina_comparison.json.

TODO: verify completed 64 steps, unchanged teacher, reload final adapter, finish 128-example post-evaluation, inspect paired changes and report accuracy difference. Earlier synthetic run is stopped and excluded.
