"""Build and inspect training subsets and recipe pools.

  python scripts/subsets.py list                      # subsets: version, stale?, rows, counts
  python scripts/subsets.py build kodcode-4o ...      # (re)build when its version changed
  python scripts/subsets.py mix configs/recipes/code/<x>.json
  python scripts/subsets.py stats kodcode-4o --by source kind
"""
import argparse, collections, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mopd.data import recipes, subsets


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    b = sub.add_parser("build"); b.add_argument("names", nargs="+")
    m = sub.add_parser("mix"); m.add_argument("recipes", nargs="+")
    s = sub.add_parser("stats"); s.add_argument("name"); s.add_argument("--by", nargs="+", default=["source"])
    a = ap.parse_args()
    if a.cmd == "list":
        for name in subsets.SUBSETS:
            path = ROOT / "data/subsets" / name / "manifest.json"
            if not path.exists():
                print(f"{name:20s} not built"); continue
            m = json.loads(path.read_text())
            stale = "STALE" if m["version"] != subsets.builder_version(name) else "ok"
            print(f"{name:20s} {m['version']} {stale:5s} {m['counts']}")
    elif a.cmd == "build":
        for name in a.names:
            print(json.dumps({k: v for k, v in subsets.build(name).items() if k != "decontam"}), flush=True)
    elif a.cmd == "mix":
        for path in a.recipes:
            m = recipes.build(path)
            print(json.dumps({k: m[k] for k in ("name", "version", "train_rows", "dev_rows", "parts")}, indent=1), flush=True)
    elif a.cmd == "stats":
        c = collections.Counter(tuple(recipes.field(r, k) for k in a.by) for r in subsets.rows(a.name))
        for key, n in c.most_common():
            print(n, *key)


if __name__ == "__main__":
    main()
