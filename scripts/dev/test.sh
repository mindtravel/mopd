#!/bin/bash
# CPU tests:  scripts/dev/test.sh            (tests/*.py; needs torch + transformers for some)
# GPU tests:  scripts/dev/test.sh gpu        (tests/gpu/*.py; run on para5090 with an idle GPU, see each file)
set -e
cd "$(dirname "$0")/../.."
PY=${PY:-$( [ -x mopd_train/bin/python ] && echo mopd_train/bin/python || echo python3 )}
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
if [ "$1" = "gpu" ]; then
  for t in tests/gpu/test_engine_stream.py tests/gpu/test_fsdp_gather.py; do echo "== $t (see file header for the launch command)"; done
  $PY -m unittest -v tests.gpu.test_dual_template tests.gpu.test_monitor_math_scoring
else
  CUDA_VISIBLE_DEVICES= $PY -m unittest discover -s tests -p "test_*.py" -t . "$@"
fi
