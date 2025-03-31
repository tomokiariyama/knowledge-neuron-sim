#!/bin/bash

DIR=$(cd $(dirname $0) && pwd)

uv run ${DIR}/../wasserstein_distance.py \
    -mn "pythia" \
    -rp "${DIR}/../work/results/original_chatgpt_data/original_similar_concepts/1token/pythia-410m-deduped" \
    -sp "${DIR}/../work/figures/wasserstein_distance" \
    --intermediate_product_save_path "${DIR}/../work/figures/wasserstein_distance/intermediate_products" \
    --cbar_log_scale

uv run ${DIR}/../wasserstein_distance.py \
    -mn "pythia" \
    -rp "${DIR}/../work/results/original_chatgpt_data/original_similar_concepts/1token/pythia-410m-deduped" \
    -sp "${DIR}/../work/figures/wasserstein_distance" \
    --intermediate_product_save_path "${DIR}/../work/figures/wasserstein_distance/intermediate_products" \
    --binary_heatmap
