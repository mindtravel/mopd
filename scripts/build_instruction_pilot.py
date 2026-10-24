"""Normalize Alpaca prompts; reserve IFEval and an independent dev split."""

import json, random, hashlib
from pathlib import Path
from mopd.data.pool import text_hash, validate_holdout


def main():
    root = Path("/ssd/lzx/mopd/data/instruction")
    alpaca = json.loads((root / "raw/alpaca_data.json").read_text())
    benchmark = [json.loads(l) for l in (root / "raw/ifeval_input_data.jsonl").read_text().splitlines() if l.strip()]
    evalrows = [
        {
            "id": str(r["key"]),
            "domain": "instruction",
            "split": "test",
            "source": "google/IFEval",
            "prompt": r["prompt"],
            "metadata": r,
        }
        for r in benchmark
    ]
    seen = {text_hash(r["prompt"]) for r in evalrows}
    rows = []
    for i, r in enumerate(alpaca):
        prompt = r["instruction"] + ("\n\n" + r["input"] if r.get("input", "").strip() else "")
        h = text_hash(prompt)
        if h in seen:
            continue
        seen.add(h)
        rows.append(
            {
                "id": "alpaca-" + str(i),
                "domain": "instruction",
                "split": "train",
                "source": "tatsu-lab/stanford_alpaca",
                "prompt": prompt,
            }
        )
    random.Random(20261013).shuffle(rows)
    dev = [dict(r, split="dev") for r in rows[:256]]
    train = rows[256:]
    validate_holdout(train, dev + evalrows)
    for name, data in [("train", train), ("dev", dev), ("ifeval", evalrows)]:
        path = root / (name + ".jsonl")
        with path.open("x") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, len(data), hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
