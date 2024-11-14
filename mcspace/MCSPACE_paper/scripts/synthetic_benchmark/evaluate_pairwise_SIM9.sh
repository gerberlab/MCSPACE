#!/bin/bash
set -e
source settings.sh

python synthetic_benchmark/helpers/pairwise/eval_ecosim_results.py -d "${PROJECT_DIR}" -o "${OUTPUT_DIR}"
