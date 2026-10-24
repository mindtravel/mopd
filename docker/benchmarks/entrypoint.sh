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
    /opt/ifbench/venv/bin/python -c "import instructions_registry; print('ifbench verifiers', len(instructions_registry.INSTRUCTION_DICT))" 2>/dev/null \
      || (cd /opt/ifbench/repo && /opt/ifbench/venv/bin/python -c "import instructions_registry as r; print('ifbench verifiers', len(r.INSTRUCTION_DICT))")
    /opt/automationbench/venv/bin/auto-bench --help >/dev/null && echo "automationbench cli ok" ;;
  *) echo "unknown benchmark $what" >&2; exit 2 ;;
esac
