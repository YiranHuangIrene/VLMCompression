#!/usr/bin/env bash
# Layer-wise pruning sweep for Bunny-v1.0-3B.
# Set BUNNY_DATA_PATH to the Bunny finetune data dir (contains bunny_695k.json and images/).
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEVICE="${DEVICE:-cuda:0}"

for N_SAMPLES in 50 100; do
  for N_PRUNE in 5 10 15 21; do
    python "$SCRIPT_DIR/short_gpt/prune.py" \
      --model_name BAAI/Bunny-v1_0-3B \
      --num_examples "$N_SAMPLES" \
      --n_prune_layers "$N_PRUNE" \
      --device "$DEVICE"
  done
done
