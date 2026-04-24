#!/bin/bash -x
#SBATCH --account=taco-vlm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --output=slurm-logs/llava-prune-0.8.%j
#SBATCH --error=slurm-errors/llava-prune-0.8.%j
#SBATCH --time=24:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:4
#SBATCH --job-name=llava-prune-0.8
# For gpus and and booster partition

# *** start of job script ***
# Note: The current working directory at this point is
# the directory where sbatch was executed.
# eval "$(conda shell.bash hook)"
# conda activate llava
export GPUS_PER_NODE=4
export SLURM_NNODES=4
# Set WANDB_API_KEY in your environment (e.g. via ~/.netrc or export) before running.
export WANDB_MODE=offline
# Optional: export HF_HOME=/path/to/hf_cache
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

python ${REPO_ROOT:-$(pwd)}/LLM-Pruner/examples/llava-vicuna_prune.py --pruning_ratio 0.6
# python ${REPO_ROOT:-$(pwd)}/LLM-Pruner/examples/llava-vicuna_prune.py --pruning_ratio 0.7
# python ${REPO_ROOT:-$(pwd)}/LLM-Pruner/examples/llava-vicuna_prune.py --pruning_ratio 0.8
# python ${REPO_ROOT:-$(pwd)}/LLM-Pruner/examples/llava-vicuna_prune.py --pruning_ratio 0.9