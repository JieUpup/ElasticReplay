#!/usr/bin/env bash
set -e

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

    python scripts/train.py \
      --method "${method}" \
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
