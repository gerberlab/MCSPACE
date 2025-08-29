#!/bin/bash
set -e
source settings.sh

python analysis/helpers/run_model_human_inulin.py -d "${PROJECT_DIR}" -o "${OUTPUT_DIR}"

# copy main results to results directory
cp -r "${OUTPUT_DIR}/analysis/human_inulin/runs/best_model" "${PROJECT_DIR}/results/analysis/human_inulin/"
cp "${OUTPUT_DIR}/analysis/human_inulin/runs/"*.csv "${PROJECT_DIR}/results/analysis/human_inulin/"
cp "${OUTPUT_DIR}/analysis/human_inulin/runs/"*.pkl "${PROJECT_DIR}/results/analysis/human_inulin/"
