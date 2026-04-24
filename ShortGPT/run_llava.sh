#!/usr/bin/env bash
# Layer-wise pruning sweep for LLaVA-v1.5-7B.
# Set LLAVA_DATA_PATH to the LLaVA mix-665k data dir.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEVICE="${DEVICE:-cuda:0}"

for N_SAMPLES in 50 100; do
  for N_PRUNE in 5 10 15 21; do
    python "$SCRIPT_DIR/short_gpt/prune.py" \
      --model_name liuhaotian/llava-v1.5-7b \
      --num_examples "$N_SAMPLES" \
      --n_prune_layers "$N_PRUNE" \
      --device "$DEVICE"
  done
done
