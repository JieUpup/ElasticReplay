#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
VISIBLE_GPUS="${CUDA_DEVICE}"
IFS=',' read -r -a GPU_LIST <<< "${VISIBLE_GPUS}"
DEVICE_IDS=""
for idx in "${!GPU_LIST[@]}"; do
  if [ -n "${DEVICE_IDS}" ]; then
    DEVICE_IDS="${DEVICE_IDS},"
  fi
  DEVICE_IDS="${DEVICE_IDS}${idx}"
done

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  echo "Run scripts/setup_server_env.sh first, or set PYTHON_BIN explicitly." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. This does not look like an NVIDIA GPU server." >&2
  exit 1
fi

echo "Using GPU ${CUDA_DEVICE}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
PYTHONPYCACHEPREFIX="${PWD}/.pycache" \
PYTHON_BIN="${PYTHON_BIN}" \
DEVICE="${DEVICE}" \
DEVICE_IDS="${DEVICE_IDS}" \
NUM_WORKERS="${NUM_WORKERS}" \
bash run_baselines.sh
