#!/bin/bash -x
#SBATCH --account=taco-vlm
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --output=./slurm-logs/Mini-InternVL-Chat-4B-V1-5_pruned_10_50_samples-ft-out.%j
#SBATCH --error=./slurm-errors/Mini-InternVL-Chat-4B-V1-5_pruned_10_50_samples-ft-err.%j
#SBATCH --time=24:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:4
#SBATCH --job-name=Mini-InternVL-Chat-4B-V1-5_pruned_10_50_samples-ft

set -x

cd "${REPO_ROOT:?set REPO_ROOT to the repo root}"/VLM/InternVL/internvl_chat
# source <conda_root>/bin/activate <env>   # activate your conda env here

# Set proxy envs here if your cluster requires them (http_proxy/https_proxy)
export GPUS_PER_NODE=1
export SLURM_NNODES=1
export WANDB__SERVICE_WAIT=300
export WANDB_HTTP_TIMEOUT=300
export WANDB_INIT_TIMEOUT=300
: "${WANDB_API_KEY:?set WANDB_API_KEY before sbatch}"
export WANDB_MODE=offline
# export HF_HOME=/path/to/hf_cache   # set if you want a shared HF cache
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901
export NCCL_IB_TIMEOUT=50
export UCX_RC_TIMEOUT=4s
export NCCL_IB_RETRY_CNT=10


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

if [ "$SYSTEMNAME" = juwelsbooster ] \
       || [ "$SYSTEMNAME" = juwels ] \
       || [ "$SYSTEMNAME" = jurecadc ] \
       || [ "$SYSTEMNAME" = jusuf ]; then
    # Allow communication over InfiniBand cells on JSC machines.
    MASTER_ADDR="$MASTER_ADDR"i
fi

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 2
# batch size per gpu: 4
# gradient accumulation steps: 2
# total batch size: 16
# epoch: 1
# python -m torch.distributed.run \
srun --jobid $SLURM_JOBID bash -c '
python -m torch.distributed.run \
  --nproc_per_node 1  \
  --nnodes 8 \
  --node_rank $SLURM_PROCID \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "OpenGVLab/Mini-InternVL-Chat-4B-V1-5" \
  --pruned_model_path "${REPO_ROOT}/ShortGPT/prune_log/Mini-InternVL-Chat-4B-V1-5_pruned_10_50_samples/pruned_model.bin" \
  --conv_style "phi3-chat" \
  --output_dir '${REPO_ROOT}/VLM/InternVL/internvl_chat/ckpt/ft-10/' \
  --meta_path "./shell/data/internvl_1_2_finetune_custom.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 12 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp False \
  --freeze_backbone True \
  --use_llm_lora 16 \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 2 \
  --learning_rate 4e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 4096 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "${REPO_ROOT}/VLM/InternVL/internvl_chat/zero_stage1_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${REPO_ROOT}/VLM/InternVL/internvl_chat/ckpt/ft-10/training_log.txt"
'