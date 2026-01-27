# Run DRPG rebuttal and its baselines

export CUDA_VISIBLE_DEVICES=0
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export NCCL_P2P_DISABLE=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTENTION
export NCCL_TIMEOUT=300
set -e

N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

base_model="openai/gpt-oss-20b"
USE_API=true  # Use API or local model
modes=("0.0" "0.1" "1.0" "2.1-0.8") # Which modes to run
encoder="/data/models/bge-m3"
planner="models/planner/test_v3/best_model.pt"


safe_model="${base_model##*/}"
if [ "$USE_API" = true ]; then
    API_FLAG="--is_api"
else
    API_FLAG=""
fi

echo "Using model: $base_model"
echo "API mode: $USE_API"
echo "Modes: ${modes[*]}"

mkdir -p data/rebuttal

# ========= Direct (0.0) =========
if [[ " ${modes[*]} " =~ " 0.0 " ]]; then
    python src/run_rebuttal.py --model "$base_model" $API_FLAG \
        --output "data/rebuttal/test_${safe_model}[0.0].json" \
        --mode 0.0 \
        --n_gpus $N_GPUS \
        --temperature 0.5
fi

# ========= Decomp (0.1) =========
if [[ " ${modes[*]} " =~ " 0.1 " ]]; then
    python src/run_rebuttal.py --model "$base_model" $API_FLAG \
        --output "data/rebuttal/test_${safe_model}[0.1].json" \
        --mode 0.1 \
        --encoder_model "$encoder" \
        --n_gpus $N_GPUS \
        --temperature 0.5
fi

# ========= DRG (1.0) =========
if [[ " ${modes[*]} " =~ " 1.0 " ]]; then
    python src/run_rebuttal.py --model "$base_model" $API_FLAG \
        --output "data/rebuttal/test_${safe_model}[1.0].json" \
        --mode 1.0 \
        --encoder_model "$encoder" \
        --n_gpus $N_GPUS \
        --temperature 0.5
fi

# ========= DRPG (2.1-0.8) =========
if [[ " ${modes[*]} " =~ " 2.1-0.8 " ]]; then
    python src/run_rebuttal.py --model "$base_model" $API_FLAG \
        --output "data/rebuttal/test_${safe_model}[2.1-0.8].json" \
        --mode 2.1 \
        --conf_threshold 0.8 \
        --planner_model "$planner" \
        --encoder_model "$encoder" \
        --n_gpus $N_GPUS \
        --temperature 0.5
fi
