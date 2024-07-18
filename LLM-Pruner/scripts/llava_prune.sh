#!/bin/bash -x
#SBATCH --account=taco-vlm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --output=/p/project/taco-vlm/huang17/VLMCompression/slurm-logs/llava-prune-0.8.%j
#SBATCH --error=/p/project/taco-vlm/huang17/VLMCompression/slurm-errors/llava-prune-0.8.%j
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
export http_proxy=http://134.94.199.178:7008
export https_proxy=$http_proxy
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$http_proxy
export GPUS_PER_NODE=4
export SLURM_NNODES=4
export WANDB_API_KEY=1882a62bfda167ae2787f2c79963b2d74f759314
export WANDB_MODE=offline
export HF_HOME=/p/scratch/taco-vlm/HF_HOME/
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

python /p/project/taco-vlm/huang17/VLMCompression/LLM-Pruner/examples/llava-vicuna_prune.py --pruning_ratio 0.6
# python /p/project/taco-vlm/huang17/VLMCompression/LLM-Pruner/examples/llava-vicuna_prune.py --pruning_ratio 0.7
# python /p/project/taco-vlm/huang17/VLMCompression/LLM-Pruner/examples/llava-vicuna_prune.py --pruning_ratio 0.8
# python /p/project/taco-vlm/huang17/VLMCompression/LLM-Pruner/examples/llava-vicuna_prune.py --pruning_ratio 0.9