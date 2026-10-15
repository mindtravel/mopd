"""Named data-mixture presets and the per-step batch plan built from them.

A domain draws its training prompts either from one `train_file` (legacy) or from a
preset in `configs/mixtures/<name>.json`:

    {"domain": "instruction", "description": "...",
     "sources": [{"name": "tulu3_if", "path": "data/instruction/tulu3_if/train.jsonl", "weight": 0.5},
                 {"name": "smol_constraints", "path": "data/instruction/smol_constraints/train.jsonl", "weight": 0.5}],
     "holdout_files": ["data/instruction/ifeval.jsonl", ...]}

Every step takes a fixed quota from each source (the domain `count` split by weight,
largest remainder), so the mixture ratio is exact per step instead of following pool
sizes. A single-source domain is planned exactly as before, so legacy runs re-plan
to the same batches.
"""
import hashlib
import json
import math
from pathlib import Path

from mopd.data.pool import build_plan, load_pool, text_hash, validate_holdout

PROJECT_ROOT = Path("/ssd/lzx/mopd")
# Source j of domain i is shuffled with seed + i + SOURCE_SEED_STRIDE * j; j=0 keeps the legacy seed.
SOURCE_SEED_STRIDE = 1000


def _absolute(path, root):
    path = Path(path)
    return str(path if path.is_absolute() else Path(root) / path)


def preset_path(name, project_root=PROJECT_ROOT):
    if name.endswith(".json") or "/" in name:
        return Path(_absolute(name, project_root))
    return Path(project_root) / "configs/mixtures" / f"{name}.json"


def load_preset(name, domain, project_root=PROJECT_ROOT):
    path = preset_path(name, project_root)
    if not path.is_file():
        raise ValueError(f"unknown mixture preset {name!r}: {path} does not exist")
    raw = path.read_bytes()
    preset = json.loads(raw)
    if preset.get("domain") != domain:
        raise ValueError(f"mixture {name!r} is for domain {preset.get('domain')!r}, not {domain!r}")
    sources = preset.get("sources") or []
    if not sources:
        raise ValueError(f"mixture {name!r} has no sources")
    names = [s.get("name") for s in sources]
    if None in names or len(set(names)) != len(names):
        raise ValueError(f"mixture {name!r}: every source needs a unique name")
    for source in sources:
        if not isinstance(source.get("weight"), (int, float)) or source["weight"] <= 0:
            raise ValueError(f"mixture {name!r}: source {source['name']!r} needs a positive weight")
    return {
        "name": name,
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "description": preset.get("description", ""),
        "sources": [{"name": s["name"], "path": _absolute(s["path"], project_root), "weight": float(s["weight"])}
                    for s in sources],
        "holdout_files": [_absolute(p, project_root) for p in preset.get("holdout_files", [])],
    }


def resolve_domain(domain, spec, project_root=PROJECT_ROOT):
    """Fill spec["sources"] (and the preset identity) in place; returns spec."""
    if "mixture" in spec and "train_file" in spec:
        raise ValueError(f"domains.{domain}: set either mixture or train_file, not both")
    if "mixture" in spec:
        preset = load_preset(spec["mixture"], domain, project_root)
        spec["mixture_preset"] = {k: preset[k] for k in ("name", "path", "sha256", "description")}
        spec["sources"] = preset["sources"]
        extra = spec.get("extra_holdout_files", [])
        spec["holdout_files"] = list(dict.fromkeys(preset["holdout_files"] + extra))
    elif "train_file" in spec:
        spec["sources"] = [{"name": Path(spec["train_file"]).parent.name, "path": spec["train_file"], "weight": 1.0}]
    else:
        raise ValueError(f"domains.{domain}: missing mixture or train_file")
    spec["source_counts"] = quotas(spec["count"], spec["sources"])
    return spec


def quotas(count, sources):
    """Split a per-step count across sources by weight; largest remainder, ties by order."""
    total = sum(s["weight"] for s in sources)
    exact = [count * s["weight"] / total for s in sources]
    result = [math.floor(x) for x in exact]
    order = sorted(range(len(sources)), key=lambda j: (-(exact[j] - result[j]), j))
    for j in order[:count - sum(result)]:
        result[j] += 1
    zero = [s["name"] for s, n in zip(sources, result) if n == 0]
    if zero:
        raise ValueError(f"per-step count {count} leaves no rows for sources {zero}; raise count or weights")
    return {s["name"]: n for s, n in zip(sources, result)}


def _rows(path, domain, source):
    return [dict(r, prompt=r.get("prompt", r.get("problem")), domain=domain, split=r.get("split", "train"),
                 mixture_source=source) for r in load_pool(path)]


def plan_domains(config):
    """Per-step batches: domains in config order, sources in preset order within each domain."""
    seed, steps = config["seed"], config["steps"]
    per_step = [[] for _ in range(steps)]
    for i, (domain, spec) in enumerate(config["domains"].items()):
        holdout = [dict(r, prompt=r.get("prompt", r.get("problem")))
                   for path in spec.get("holdout_files", []) for r in load_pool(path)]
        ids, texts = set(), set()
        for j, source in enumerate(spec["sources"]):
            rows = _rows(source["path"], domain, source["name"])
            keys = {(r["domain"], r["id"]) for r in rows}
            hashes = {text_hash(r["prompt"]) for r in rows}
            if keys & ids or hashes & texts:
                raise ValueError(f"domains.{domain}: source {source['name']!r} repeats IDs or prompts of an earlier source")
            ids |= keys
            texts |= hashes
            if holdout:
                validate_holdout(rows, holdout)
            plan = build_plan(rows, seed=seed + i + SOURCE_SEED_STRIDE * j, steps=steps,
                              batch_size=spec["source_counts"][source["name"]])
            for step, batch in enumerate(plan):
                per_step[step].extend(batch)
    return per_step
