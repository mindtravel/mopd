#!/usr/bin/env bash
# Evaluate every checkpoint of one run on the given GPUs (all669, plain_qa_v1, greedy 4096).
# Finished run: checkpoints are split round-robin, one detached eval_queue per GPU.
# Still training: one queue on the first GPU that also waits for new checkpoints.
# usage: launch_eval.sh RUN_NAME GPU[,GPU...]
set -eo pipefail
[[ $# -eq 2 ]] || { echo "usage: $0 RUN_NAME GPU[,GPU...]" >&2; exit 2; }
run_name=$1
[[ $run_name =~ ^[A-Za-z0-9._-]+$ ]] || { echo "bad run name: $run_name" >&2; exit 2; }
[[ $2 =~ ^[0-7](,[0-7])*$ ]] || { echo "bad GPU list: $2" >&2; exit 2; }
IFS=, read -ra gpus <<<"$2"
for g in "${gpus[@]}"; do
  [[ $g != 1 ]] || { echo "GPU 1 is reserved" >&2; exit 2; }
done
root=/ssd/lzx/mopd
cd "$root"
source scripts/env.sh
run=$root/runs/$run_name
manifest=$run/model-version.json
[[ -f $manifest ]] || { echo "no model-version.json in $run" >&2; exit 1; }
protocol=all669-plainqa-greedy4096-v1
data=$root/data/full_eval/all.jsonl
state=$(python -c 'import json,sys;print(json.load(open(sys.argv[1])).get("state",""))' "$run/status.json")

start_queue() {  # GPU, extra args...
  local g=$1; shift
  local log=$root/logs/eval-$run_name-gpu$g.log
  [[ ! -e $log ]] || log=$root/logs/eval-$run_name-gpu$g-$(date +%m%d%H%M%S).log
  setsid nohup python -u scripts/eval_queue.py --device "$g" --data "$data" \
    --protocol-id "$protocol" "$@" >"$log" 2>&1 </dev/null &
  echo "gpu$g pid=$! log=$log"
}

if [[ $state != complete ]]; then
  start_queue "${gpus[0]}" --watch "$run"
  exit 0
fi
mapfile -t ckpts < <(find "$run" -maxdepth 1 -type d -name 'checkpoint-*' | sort)
[[ ${#ckpts[@]} -gt 0 ]] || { echo "no checkpoints in $run" >&2; exit 1; }
for i in "${!gpus[@]}"; do
  args=()
  for j in "${!ckpts[@]}"; do
    (( j % ${#gpus[@]} == i )) || continue
    [[ -f ${ckpts[j]}/../evaluations/$(basename "${ckpts[j]}")/$protocol/summary.json ]] && continue
    args+=(--fixed "$manifest::${ckpts[j]}::plain_qa_v1")
  done
  [[ ${#args[@]} -gt 0 ]] && start_queue "${gpus[i]}" "${args[@]}"
done
exit 0
