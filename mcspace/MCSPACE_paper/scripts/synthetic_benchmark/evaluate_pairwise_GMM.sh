#!/bin/bash
set -e
source settings.sh

python synthetic_benchmark/helpers/pairwise/eval_pairwise_GMM.py -d "${PROJECT_DIR}" -o "${OUTPUT_DIR}"
