# Train the planner model

export CUDA_VISIBLE_DEVICES=0
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export NCCL_P2P_DISABLE=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTENTION
export NCCL_TIMEOUT=300
export WANDB_PROJECT=perspective_planner
set -e
N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

# For simplicity, the data for planner training is fixed in "train_planner.py".
python src/planner/train_planner.py \
    --encoder "/data/models/bge-m3" \
    --output "models/planner/66666" \
    --num_epochs 3 \
    --learning_rate 5e-5 \
    --mlp_hidden_size 2048 1024 512 \
    --log_interval 20 \
    --batch_size 32 \
    --freeze_encoder
