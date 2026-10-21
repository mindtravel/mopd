#!/usr/bin/env bash
MOPD_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export MOPD_ROOT
export HF_HOME="$MOPD_ROOT/cache/huggingface"
export PIP_CACHE_DIR="$MOPD_ROOT/cache/pip"
export TORCH_EXTENSIONS_DIR="$MOPD_ROOT/cache/torch_extensions"
export TRITON_CACHE_DIR="$MOPD_ROOT/cache/triton"
export XDG_CACHE_HOME="$MOPD_ROOT/cache/xdg"
export VLLM_CACHE_ROOT="$MOPD_ROOT/cache/vllm"
export TMPDIR="$MOPD_ROOT/tmp"
export PYTHONPATH="$MOPD_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
source "$MOPD_ROOT/mopd_train/bin/activate"
