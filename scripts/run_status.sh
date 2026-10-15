#!/usr/bin/env bash
# Read-only snapshot: run state, last metrics, rollout health, eval summaries, GPUs, MOPD processes.
# usage: run_status.sh [RUN_NAME]   (default: most recently modified run)
set -eo pipefail
root=/ssd/lzx/mopd
cd "$root"
if [[ $# -ge 1 ]]; then
  [[ $1 =~ ^[A-Za-z0-9._-]+$ ]] || { echo "bad run name: $1" >&2; exit 2; }
  run=runs/$1
else
  run=$(ls -td runs/*/ | head -1); run=${run%/}
fi
[[ -d $run ]] || { echo "no such run: $run" >&2; exit 1; }
date "+%m-%d %H:%M:%S"
./mopd_train/bin/python -I - "$run" <<'EOF'
import json,sys,glob,os,re,collections
run=sys.argv[1]
print("run",os.path.basename(run))
st=json.load(open(f"{run}/status.json")) if os.path.exists(f"{run}/status.json") else {}
print("status",{k:st[k] for k in ("state","step","steps","phase","domain") if k in st})
m=[json.loads(l) for l in open(f"{run}/metrics.jsonl")] if os.path.exists(f"{run}/metrics.jsonl") else []
if m:
    last=m[-1]
    print("steps_logged",len(m),"last_step_seconds",round(last.get("seconds",0)))
    for d,v in last.get("domains",{}).items():
        print(f"  {d}: sampled_log_ratio={v['sampled_log_ratio']:.3f} truncated={v['truncated']}/{v['count']} grad_l2={v['grad_contribution_l2']:.3f}")
odd=re.compile(r"[֐-ۿ가-힯Ѐ-ӿąęłńśźżğşı]")
if os.path.exists(f"{run}/rollouts.jsonl"):
    c=collections.defaultdict(lambda:[0,0,0])
    for l in open(f"{run}/rollouts.jsonl"):
        r=json.loads(l);k=r["domain"];c[k][0]+=1;c[k][1]+=bool(r["truncated"])
        c[k][2]+=bool(odd.search(r["response"]) and not odd.search(r["prompt"]))
    print("rollouts n/truncated/foreign_script",dict(c))
for p in sorted(glob.glob(f"{run}/evaluations/*/*/summary.json")):
    d=json.load(open(p));parts=p.split("/")
    row=[f"{dom}={v['accuracy']:.4f}(trunc {v['truncated']})" for dom,v in d["domains"].items()]
    print("eval",parts[-3],parts[-2]," ".join(row))
for p in sorted(glob.glob(f"{run}/evaluations/*/*/generation-progress.json")):
    if not os.path.exists(p.replace("generation-progress.json","summary.json")):
        g=json.load(open(p));print("eval-running",p.split("/")[-3],f"{g.get('done')}/{g.get('total')}")
EOF
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
pgrep -af 'train_multidomain_opd|eval_queue|evaluate_model_version' | grep -v pgrep | cut -c1-140 || echo "no MOPD processes"
