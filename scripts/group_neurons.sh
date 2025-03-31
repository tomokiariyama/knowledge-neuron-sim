#!/bin/bash

DIR=$(cd $(dirname $0) && pwd)

uv run ${DIR}/../group_neurons.py \
    -mn "pythia" \
    -rp "${DIR}/../work/results/original_chatgpt_data/original_similar_concepts/1token/pythia-410m-deduped" \
    -sp "${DIR}/../work/figures/group_neurons" \
    --intermediate_product_save_path "${DIR}/../work/figures/group_neurons/intermediate_products"