#!/bin/bash -x
#SBATCH --account=your-account
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --output=slurm-logs/internvl-prune.%j
#SBATCH --error=slurm-errors/internvl-prune.%j
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=internvl-prune

# Activate your Python env (edit to match your setup)
# eval "$(conda shell.bash hook)"
# conda activate vlmc

# Set WANDB_API_KEY in your environment before running (not committed to repo).
export WANDB_MODE=offline
# Optional: export HF_HOME=/path/to/hf_cache

BASE_MODEL="${BASE_MODEL:-OpenGVLab/Mini-InternVL-Chat-4B-V1-5}"
PRUNING_RATIO="${PRUNING_RATIO:-0.25}"
NUM_EXAMPLES="${NUM_EXAMPLES:-10}"
DATASET="${DATASET:-c4}"

python "${REPO_ROOT:-$(pwd)}/LLM-Pruner/examples/InternVL.py" \
    --base_model "${BASE_MODEL}" \
    --pruning_ratio "${PRUNING_RATIO}" \
    --num_examples "${NUM_EXAMPLES}" \
    --dataset "${DATASET}" \
    --save_model
