# Do pairwise comparison for a list of rebuttals

export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
set -e

names=("llama-70B[1.0]" "llama-70B[2.c]" "llama-70B[2.j]" "llama-70B[2.1-0.8]") 

for ((i=0; i<${#names[@]}; i++)); do
    for ((j=i+1; j<${#names[@]}; j++)); do
        
        name1="${names[i]}"
        name2="${names[j]}"

        out_file="/data/ph16/Graph_of_Persuasion/data/comparative_score_analysis/test_${name1}_${name2}_gpt.json"

        if [ -f "$out_file" ]; then
            echo "Skipping: $name1 vs $name2  (output exists: $out_file)"
            continue
        fi

        echo "Comparing: $name1 vs $name2"
        
        python src/compare_rebuttals.py \
            --model gpt-4o \
            --file_1 /data/ph16/Graph_of_Persuasion/data/rebuttal/test_${name1}.json \
            --file_2 /data/ph16/Graph_of_Persuasion/data/rebuttal/test_${name2}.json \
            --output "$out_file"

    done
done
