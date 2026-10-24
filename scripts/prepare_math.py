"""Download MATH train/test concurrently; freeze disjoint prompt manifests."""

import concurrent.futures as cf
import hashlib, json, random, time
from pathlib import Path
import requests
import pyarrow.parquet as pq

ROOT = Path("/ssd/lzx/mopd/data/math")
REPO = "EleutherAI/hendrycks_math"
BASE = "https://hf-mirror.com"


def get(url):
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=90)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    rev = get(f"{BASE}/api/datasets/{REPO}").json()["sha"]
    files = get(f"{BASE}/api/datasets/{REPO}/tree/{rev}?recursive=true").json()
    files = [f for f in files if f["path"].endswith(".parquet")]

    def download(f):
        p = ROOT / "raw" / f["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        data = get(f"{BASE}/datasets/{REPO}/resolve/{rev}/{f['path']}?download=true").content
        sha = hashlib.sha256(data).hexdigest()
        assert sha == f["lfs"]["oid"], f["path"]
        p.write_bytes(data)
        print("downloaded", f["path"], len(data), flush=True)
        return {"path": f["path"], "sha256": sha}

    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        manifest = list(pool.map(download, files))
    train = []
    test = []
    for f in sorted(files, key=lambda x: x["path"]):
        split = "train" if "/train-" in f["path"] else "test"
        for i, r in enumerate(pq.read_table(ROOT / "raw" / f["path"]).to_pylist()):
            r["id"] = f"{f['path']}:{i}"
            (train if split == "train" else test).append(r)
    key = lambda r: " ".join(r["problem"].split())
    test_keys = {key(r) for r in test}
    eligible = [
        r
        for r in train
        if r["level"] in ["Level 3", "Level 4", "Level 5"] and "[asy]" not in r["problem"] and key(r) not in test_keys
    ]
    random.Random(20261013).shuffle(eligible)
    # training receives no reference solutions.
    training = [{k: r[k] for k in ["id", "problem", "level", "type"]} for r in eligible[:1024]]
    pool = [r for r in test if "[asy]" not in r["problem"]]
    random.Random(8171).shuffle(pool)
    evaluation = pool[:128]
    for name, rows in [("train", training), ("eval", evaluation)]:
        with (ROOT / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {
        "repo": REPO,
        "revision": rev,
        "files": manifest,
        "train_total": len(train),
        "test_total": len(test),
        "train_selected": len(training),
        "eval_selected": len(evaluation),
        "overlap": len({key(r) for r in training} & {key(r) for r in evaluation}),
        "train_policy": "Level 3-5; exclude Asymptote diagrams and exact normalized test overlap; seed 20261013",
        "eval_policy": "128 fixed random official test problems excluding Asymptote diagrams; seed 8171; not full MATH",
        "eval_levels": {
            level: sum(r["level"] == level for r in evaluation)
            for level in ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]
        },
    }
    (ROOT / "manifest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
