#!/usr/bin/env bash
# Launch one multidomain training run detached from the shell.
# Refuses to reuse a run directory, log or PID file; records the PID by absolute path.
set -eo pipefail
[[ $# -eq 1 ]] || { echo "usage: $0 CONFIG.json" >&2; exit 2; }
root=/ssd/lzx/mopd
config=$(readlink -f "$1")
cd "$root"
source scripts/env.sh
output=$(./mopd_train/bin/python -c 'import json,sys;print(json.load(open(sys.argv[1]))["output"])' "$config")
name=$(basename "$output")
log=$root/logs/$name.log
pidfile=$root/logs/$name.pid
for path in "$output" "$log" "$pidfile"; do
  [[ ! -e $path ]] || { echo "already exists: $path" >&2; exit 1; }
done
setsid nohup ./mopd_train/bin/python -u scripts/train_multidomain_opd.py --config "$config" >"$log" 2>&1 </dev/null &
echo $! >"$pidfile"
echo "pid=$(cat "$pidfile") log=$log"
