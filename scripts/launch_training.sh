#!/usr/bin/env bash
# Launch one multidomain training run detached from the shell, plus its evaluation.
# Refuses to reuse a run directory, log or PID file; records the PID by absolute path.
# Evaluation: GPUs in the config's optional "eval_devices" that training does not use start
# a watching eval queue immediately; otherwise a detached waiter runs launch_eval.sh on the
# run's own GPUs once the run reaches a final state, so checkpoints never sit unevaluated.
set -eo pipefail
[[ $# -eq 1 ]] || { echo "usage: $0 CONFIG.json" >&2; exit 2; }
root=/ssd/lzx/mopd
config=$(readlink -f "$1")
cd "$root"
source scripts/env.sh
read -r output train_gpus eval_gpus < <(./mopd_train/bin/python - "$config" <<'EOF'
import json,sys
c=json.load(open(sys.argv[1]))
train={c["student_device"],*c.get("rollout_devices",[]),*(d["device"] for d in c["domains"].values())}
idx=lambda ds:sorted({d.replace("cuda:","") for d in ds}-{"1"})
free=[g for g in idx(c.get("eval_devices",[])) if f"cuda:{g}" not in train]
print(c["output"],",".join(idx(train)),",".join(free) or "-")
EOF
)
name=$(basename "$output")
log=$root/logs/$name.log
pidfile=$root/logs/$name.pid
for path in "$output" "$log" "$pidfile"; do
  [[ ! -e $path ]] || { echo "already exists: $path" >&2; exit 1; }
done
setsid nohup ./mopd_train/bin/python -u scripts/train_multidomain_opd.py --config "$config" >"$log" 2>&1 </dev/null &
pid=$!
echo $pid >"$pidfile"
echo "pid=$pid log=$log"

if [[ $eval_gpus != - ]]; then
  # The run directory appears once training has validated its plan; launch_eval.sh needs it.
  for _ in $(seq 120); do [[ -f $output/model-version.json ]] && break; sleep 5; done
  scripts/launch_eval.sh "$name" "$eval_gpus"
else
  waiter_log=$root/logs/$name.eval-waiter.log
  setsid nohup bash -c '
    while kill -0 '"$pid"' 2>/dev/null; do sleep 60; done
    state=$(python -c "import json,sys;print(json.load(open(sys.argv[1])).get(\"state\",\"\"))" "'"$output"'/status.json" 2>/dev/null || true)
    echo "training pid '"$pid"' exited, state=$state"
    [[ $state == complete ]] && exec '"$root"'/scripts/launch_eval.sh "'"$name"'" "'"$train_gpus"'"
    echo "not evaluating: run did not complete"' >"$waiter_log" 2>&1 </dev/null &
  echo "eval waiter pid=$! gpus=$train_gpus log=$waiter_log"
fi
