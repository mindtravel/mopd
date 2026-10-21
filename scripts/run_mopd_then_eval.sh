#!/usr/bin/env bash
# Train one MOPD config, then evaluate its final checkpoint on one protocol and refresh the
# comparison table. Detach with setsid; progress in logs/<run>.log and logs/<run>.eval.log.
set -euo pipefail
[[ $# -ge 2 ]] || { echo "usage: $0 CONFIG.json EVAL_GPU [PROTOCOL_ID]" >&2; exit 2; }
root=/ssd/lzx/mopd; cd "$root"; source scripts/env.sh
config=$(readlink -f "$1"); gpu=$2; protocol=${3:-unified-imc-plainqa-greedy4096-v1}
read -r output steps < <(python -c "import json,sys;c=json.load(open(sys.argv[1]));print(c['output'],c['steps'])" "$config")
name=$(basename "$output")
python -u scripts/train_multidomain_opd.py --config "$config" > "logs/$name.log" 2>&1
python scripts/evaluate_model_version.py --model-version "$output/model-version.json" \
  --checkpoint "$(printf 'checkpoint-%04d' "$steps")" --data data/full_eval/unified-v1/all.jsonl \
  --protocol-id "$protocol" --device "$gpu" --max-new-tokens 4096 --batch-size 64 --max-seqs 64 > "logs/$name.eval.log" 2>&1
python scripts/collect_sft_comparison.py >> "logs/$name.eval.log" 2>&1
echo DONE >> "logs/$name.eval.log"
