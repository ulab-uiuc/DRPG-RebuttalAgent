# Use judge model to evaluate a list of rebuttals

export CUDA_VISIBLE_DEVICES=0,1,2,3
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export NCCL_P2P_DISABLE=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export NCCL_TIMEOUT=100
set -e

N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

judge_model="xxx"
rebuttal_files=(
  "llama-70B[1.0]"
  "llama-70B[2.1-0.8]"
)

for file in "${rebuttal_files[@]}"; do
    echo "$file"
    python src/run_reviewer.py \
        --input data/rebuttal/test_"$file".json \
        --output data/revised_score/test_"$file"_Judge_re.json \
        --model "$judge_model" \
        --n_gpus $N_GPUS
done
