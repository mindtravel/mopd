from mopd import paths
import json, re, random, hashlib, collections, difflib
from pathlib import Path
import pyarrow.parquet as pq

root = paths.ROOT
norm = lambda x: re.sub(r"[^a-z0-9]", "", x.lower())
test = []
for p in (root / "data/math/raw").glob("*/test-*.parquet"):
    test.extend(pq.read_table(p).to_pylist())
keys = {norm(r["problem"]) for r in test}


def grams(s):
    return {s[i : i + 5] for i in range(max(0, len(s) - 4))}


testnorm = [norm(r["problem"]) for r in test]
tg = [grams(s) for s in testnorm]
inv = collections.defaultdict(list)
for i, g in enumerate(tg):
    for v in g:
        inv[v].append(i)
rows = pq.read_table(root / "data/numina/train-00000.parquet", columns=["source", "problem"]).to_pylist()
eligible = [
    (i, r)
    for i, r in enumerate(rows)
    if r["source"] in ["olympiads", "amc_aime"] and "[asy]" not in r["problem"] and 80 <= len(r["problem"]) <= 3000
]
random.Random(20261013).shuffle(eligible)
selected = []
seen = set()
excluded = collections.Counter()
for i, r in eligible:
    s = norm(r["problem"])
    if s in seen or s in keys:
        excluded["exact"] += 1
        continue
    g = grams(s)
    counts = collections.Counter(j for v in g for j in inv.get(v, []))
    near = False
    for j, count in counts.most_common(5):
        jac = count / max(1, len(g | tg[j]))
        ratio = difflib.SequenceMatcher(None, s, testnorm[j], autojunk=False).ratio()
        if jac >= 0.65 or ratio >= 0.8:
            near = True
            break
    if near:
        excluded["near"] += 1
        continue
    seen.add(s)
    selected.append({"id": f"numina-shard0:{i}", **r})
    if len(selected) == 1024:
        break
assert len(selected) == 1024
out = root / "data/numina/train.jsonl"
out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected))
report = {
    "selected": len(selected),
    "sources": dict(collections.Counter(r["source"] for r in selected)),
    "excluded": dict(excluded),
    "benchmark_checked": len(test),
    "train_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    "dedup": "alphanumeric normalization exact; top5 character-5gram candidates Jaccard>=0.65 or SequenceMatcher>=0.8",
    "limitation": "heuristic near-duplicate screen; cannot exclude semantic paraphrases or model pretraining contamination",
}
(root / "data/numina/selection.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
c = json.loads((root / "configs/math_opd.json").read_text())
c.update(train_file=str(out), max_new_tokens=1024, output=str(root / "runs/20261013-numina-competition-opd"))
(root / "configs/numina_opd.json").write_text(json.dumps(c, indent=2))
