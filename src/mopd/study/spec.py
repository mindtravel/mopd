"""Load a study and expand it into runs.

  study: full-sft-lr                    # name: results/studies/<name>, runs/studies/<name>
  description: ...
  base: {method: sft, lr: 2e-6, ...}    # differences from the defaults (mopd.study.resolve.DEFAULTS)
  axes: {domain: [math, code], lr: [5e-7, 2e-6]}       # grid: every combination is one run
  name: "{domain}-lr{lr}"               # run name template over the run's fields
  data: {math: math-numina-v3, code: code-rstar-instr11}   # domain -> registry dataset
  evals: [unified-v3, {suite: lcbdev-v3, if: "domain == 'code'"}]
  checkpoints: [64, 128, 256]
  teachers: {math: {study: full-sft-lr, run: math-lr2e-06, checkpoint: 256}}   # method: mopd

`if:` conditions are expressions over the run's fields: names, constants, comparisons, `in`,
`and` / `or` / `not` -- nothing else is evaluated.
"""

import ast
import itertools
from pathlib import Path

import yaml

ALLOWED = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.Compare, ast.Eq, ast.NotEq,
           ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Name, ast.Load, ast.Constant, ast.List, ast.Tuple)


def condition(expr, fields):
    """Evaluate a restricted expression over a run's fields."""
    tree = ast.parse(expr, mode="eval")
    bad = [type(n).__name__ for n in ast.walk(tree) if not isinstance(n, ALLOWED)]
    if bad:
        raise ValueError(f"condition {expr!r}: {sorted(set(bad))} not allowed")
    unknown = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} - set(fields)
    if unknown:
        raise ValueError(f"condition {expr!r}: unknown fields {sorted(unknown)}")
    return bool(eval(compile(tree, "<if>", "eval"), {"__builtins__": {}}, dict(fields)))


def load(path):
    study = yaml.safe_load(Path(path).read_text())
    for key in ("study", "base"):
        if key not in study:
            raise ValueError(f"{path}: missing {key!r}")
    unknown = set(study) - {"study", "description", "base", "axes", "name", "data", "evals", "checkpoints", "teachers", "baseline"}
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    return study


def expand(study):
    """[(run name, fields)] in grid order: base overridden by each combination of the axes."""
    axes = study.get("axes") or {}
    overlap = set(axes) & set(study["base"])
    if overlap:
        raise ValueError(f"{study['study']}: {sorted(overlap)} both in base and axes")
    runs, names = [], set()
    for combo in itertools.product(*axes.values()) if axes else [()]:
        fields = {**study["base"], **dict(zip(axes, combo))}
        name = study.get("name", "run").format(**fields)
        if name in names:
            raise ValueError(f"{study['study']}: run name {name!r} repeats; put every axis in `name`")
        names.add(name)
        runs.append((name, fields))
    return runs


def evals_for(study, fields):
    """The eval suite names that apply to this run."""
    out = []
    for e in study.get("evals") or []:
        e = {"suite": e} if isinstance(e, str) else e
        if "if" not in e or condition(e["if"], fields):
            out.append(e["suite"])
    return out
