#!/usr/bin/env bash
set -e

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DEVICE="${DEVICE:-auto}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE_IDS="${DEVICE_IDS:-}"

SEEDS=(42 43 44)

METHODS=(
  finetune
  full_replay
  selective_replay
)

for seed in "${SEEDS[@]}"; do
  for method in "${METHODS[@]}"; do
    echo "========================================"
    echo "Running method=${method}, seed=${seed}"
    echo "========================================"

    "${PYTHON_BIN}" scripts/train.py \
      --method "${method}" \
      --device "${DEVICE}" \
      --device_ids "${DEVICE_IDS}" \
      --num_workers "${NUM_WORKERS}" \
      --seed "${seed}" \
      --num_tasks 5 \
      --samples_per_task 1000 \
      --T 12 \
      --N 64 \
      --F 16 \
      --num_classes 2 \
      --hidden_dim 128 \
      --batch_size 32 \
      --epochs 20 \
      --lr 1e-3 \
      --buffer_size 500 \
      --replay_batch_size 32 \
      --topk_windows 2 \
      --noise_std 0.15 \
      --task_shift 0.8 \
      --class_signal 1.2
  done
done
