"""Replay one recorded MOPD step through the refactored scorer to check the streaming pipeline.

Loads the run's student at checkpoint k and its teachers, rebuilds step k+1's trajectories from
rollouts.jsonl (prompt prefix from batch_plan.json, recorded response token IDs), and runs
DomainBatcher + OPDStepScorer:
  - in plan order, twice: must reproduce the per-domain statistics the training run logged for
    step k+1 (refactor fidelity) and gives the same-shape noise level;
  - in a shuffled order (what streaming produces): the gradient may differ only at the
    micro-batch-composition noise level, statistics only in the last digits.

  python scripts/verify_streaming_equivalence.py --run runs/<mopd run> --checkpoint checkpoint-0064
"""

import argparse, io, json, random
from pathlib import Path
import torch
from mopd.backends.multi_teacher import load_models
from mopd.data.multidomain import domain_scale
from mopd.protocol import student_prompt
from mopd.training.batching import generation_eos_ids
from mopd.training.opd_step import OPDStepScorer
from mopd.training.streaming import DomainBatcher


def replay(c, student, stok, teachers, params, eos, trajs, step):
    for p in params:
        p.grad = None
    weights = {d: s["weight"] for d, s in c["domains"].items()}
    counts = {d: s["count"] for d, s in c["domains"].items()}
    scales = {d: domain_scale(d, weights, counts) for d in counts}
    sc = OPDStepScorer(
        config=c,
        student=student,
        student_tokenizer=stok,
        teachers=teachers,
        params=params,
        scales=scales,
        eos_ids=eos,
        sync=lambda: None,
        step=step,
    )
    planned = {d: sum(t["row"]["domain"] == d for t in trajs) for d in c["domains"]}
    for d, chunk in DomainBatcher(c["micro_batch_size"], planned).consume(trajs):
        sc.process(d, chunk)
    return torch.cat([p.grad.detach().float().flatten() for p in params]), sc.domain_stats()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run = Path(a.run)
    k = int(a.checkpoint.rsplit("-", 1)[1])
    step = k + 1
    c = json.loads((run / "config.json").read_text())
    for f in sorted(run.glob("resume-*.json")):  # runtime overrides in force at that step
        r = json.loads(f.read_text())
        if int(f.stem.split("-")[1]) < step:
            c.update(r["overrides"])
    c = {**c, "student_adapter": str(run / a.checkpoint)}
    stok, student, teachers = load_models(c)
    eos = generation_eos_ids(student, stok)
    params = [p for p in student.parameters() if p.requires_grad]
    plan = json.loads((run / "batch_plan.json").read_text())[step - 1]
    pos = {r["id"]: i for i, r in enumerate(plan)}
    logged = [json.loads(l) for l in open(run / "rollouts.jsonl") if json.loads(l)["step"] == step]
    trajs = sorted(
        [
            {
                "index": pos[r["id"]],
                "row": plan[pos[r["id"]]],
                "answer": r["response_ids"],
                "prefix": stok.encode(student_prompt(stok, plan[pos[r["id"]]], c["student_prompt_format"])),
            }
            for r in logged
        ],
        key=lambda t: t["index"],
    )
    metrics = next(json.loads(l) for l in open(run / "metrics.jsonl") if json.loads(l)["step"] == step)
    g1, s1 = replay(c, student, stok, teachers, params, eos, trajs, step)
    g2, _ = replay(c, student, stok, teachers, params, eos, trajs, step)
    shuffled = trajs[:]
    random.Random(a.seed).shuffle(shuffled)
    g3, s3 = replay(c, student, stok, teachers, params, eos, shuffled, step)
    rel = lambda x, y: float((x - y).norm() / y.norm())
    out = {
        "step": step,
        "micro_batch_size": c["micro_batch_size"],
        "rel_l2 in-order repeat (same-shape noise)": rel(g2, g1),
        "rel_l2 shuffled vs in-order": rel(g3, g1),
        "cos shuffled vs in-order": float(torch.nn.functional.cosine_similarity(g3, g1, dim=0)),
        "per_domain": {},
    }
    for d in c["domains"]:
        L = metrics["domains"][d]
        out["per_domain"][d] = {
            k: {"logged": L[k], "replay": s1[d][k], "shuffled": s3[d][k]}
            for k in ("loss", "sampled_log_ratio", "weighted_loss", "grad_contribution_l2", "tokens", "count")
        }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
