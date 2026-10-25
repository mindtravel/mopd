"""Score breakdowns of graded benchmark outputs, shared by leaderboards and scripts/report.py.

lcb_breakdown: LiveCodeBench-style rows (metadata.testtype / difficulty) -> all, functional, stdin,
easy, medium, hard (percent correct), lenient, truncated count, rows. opd_step_summary: medians of an
on-policy distillation run's per-step timings and utilization (metrics.jsonl).
"""

import json
import statistics
from pathlib import Path


def load_rows(data):
    return {json.loads(l)["id"]: json.loads(l) for l in Path(data).read_text().splitlines() if l.strip()}


def lcb_breakdown(cases_path, rows):
    """rows: {id: benchmark row} (load_rows)."""
    cases = [json.loads(l) for l in Path(cases_path).read_text().splitlines() if l.strip()]
    meta = lambda c: rows[c["id"]].get("metadata", {})
    pct = lambda xs: round(100 * sum(c["correct"] for c in xs) / len(xs), 1) if xs else None
    out = {"all": pct(cases)}
    for key, field in (
        ("functional", "testtype"),
        ("stdin", "testtype"),
        ("easy", "difficulty"),
        ("medium", "difficulty"),
        ("hard", "difficulty"),
    ):
        out[key] = pct([c for c in cases if meta(c).get(field) == key])
    out["lenient"] = round(100 * sum(c.get("correct_lenient", c["correct"]) for c in cases) / len(cases), 1)
    out["truncated"] = sum(bool(c.get("truncated")) for c in cases)
    out["rows"] = len(cases)
    return out


def opd_step_summary(metrics_path):
    m = [json.loads(l) for l in Path(metrics_path).read_text().splitlines() if l.strip()]
    med = lambda f: round(statistics.median(f(x) for x in m), 3)
    out = {"steps": len(m), "seconds": med(lambda x: x["seconds"])}
    for k in ("publish_seconds", "generation_seconds", "teacher_seconds", "student_seconds", "optimizer_seconds"):
        if k in m[0]:
            out[k] = med(lambda x, k=k: x[k])
    if "utilization" in m[0]:
        for k in m[0]["utilization"]:
            if isinstance(m[0]["utilization"][k], (int, float)):
                out[k] = med(lambda x, k=k: x["utilization"][k])
    out["student_peak_gib"] = round(max(x.get("student_peak_gib", 0) for x in m), 1)
    return out
