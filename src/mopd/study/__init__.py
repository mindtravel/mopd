"""Studies: one comparison = a base configuration plus axes expanded as a grid, written as the
differences from the defaults (configs/studies/<name>.yaml). Replaces copying experiment specs per
variant; everything else is derived:

  registry   names -> paths and sha256 pins (configs/registry.yaml): models, data, eval suites
  spec       load a study, expand its axes, evaluate `if:` conditions over the axis values
  resolve    one run -> the training config the trainers already read (scripts/train_sft.py),
             its identity (sha of the fields that change results) and its run directory
  runner     the job graph on the scheduler: training on the "train" GPUs, evaluations on "eval";
             a run whose identity already finished anywhere under runs/ is reused, not retrained

  python scripts/study.py show configs/studies/full-sft-lr.yaml     # runs, identities, reuse
  python scripts/study.py run  configs/studies/full-sft-lr.yaml
"""
