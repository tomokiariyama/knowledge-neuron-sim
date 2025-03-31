#!/bin/bash

MODEL_NAME="pythia-410m-deduped"

CHECKPOINTS=(0 512 1000 3000 5000 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 110000 120000 130000 140000 143000)
NUM_CHECKPOINTS=${#CHECKPOINTS[@]}

DATE=$(date '+%F_%T')

DIR=$(cd $(dirname $0) && pwd)

for i in $(seq $((NUM_CHECKPOINTS))); do
    CHECKPOINT=${CHECKPOINTS[i-1]}
    echo "Running experiment of ${MODEL_NAME} for checkpoint ${CHECKPOINT}"

    uv run ${DIR}/../evaluate_attribution_scores.py \
    --model_name "EleutherAI/${MODEL_NAME}" \
    --training_step ${CHECKPOINT} \
    --dataset_type original_similar_concepts \
    --dataset_split 1token \
    --dataset_path "${DIR}/../data/generated_data_1token.json" \
    --number_of_templates 4 \
    --adaptive_threshold -10.0 \
    --max_words 30 \
    --logfile_name "${MODEL_NAME}_step${CHECKPOINT}_${DATE}" \
    --local_rank 0 \
    --save_path "${DIR}/../work/results/original_chatgpt_data"
done
