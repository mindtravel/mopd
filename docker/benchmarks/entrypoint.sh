#!/usr/bin/env bash
# bench ifbench <ifbench_cli args...> | bench automationbench <auto-bench args...> | bench selftest
# Results go under /out (mounted from the host).
set -euo pipefail
what="${1:?usage: bench ifbench|automationbench|selftest ...}"; shift
case "$what" in
  ifbench)
    # runs/<name> lives in /out so results survive the container
    mkdir -p /out/ifbench && ln -sfn /out/ifbench /opt/ifbench/runs
    exec /opt/ifbench/venv/bin/python /opt/ifbench/scripts/ifbench_cli.py "$@" ;;
  automationbench)
    cd /opt/automationbench/repo
    exec /opt/automationbench/venv/bin/auto-bench "$@" ;;
  selftest)
    # official sample answers must score as in the native deployment (strict prompt accuracy 0.28)
    out="$(mktemp -d)"; cd /opt/ifbench/repo
    /opt/ifbench/venv/bin/python run_eval.py --input_data=data/IFBench_test.jsonl \
      --input_response_data=data/sample_output.jsonl --output_dir="$out" >/dev/null 2>&1
    /opt/ifbench/venv/bin/python - "$out" <<'PY'
import json, sys, pathlib
rows = [json.loads(l) for l in (pathlib.Path(sys.argv[1]) / "eval_results_strict.jsonl").open()]
acc = sum(r["follow_all_instructions"] for r in rows) / len(rows)
print(f"ifbench sample strict prompt accuracy {acc:.4f} over {len(rows)}")
assert len(rows) == 300 and abs(acc - 0.28) < 1e-9
PY
    /opt/automationbench/venv/bin/auto-bench --help >/dev/null && echo "automationbench cli ok" ;;
  *) echo "unknown benchmark $what" >&2; exit 2 ;;
esac
