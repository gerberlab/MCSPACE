#!/bin/bash
set -e
source settings.sh

python synthetic_benchmark/helpers/pairwise/run_fisher_pairwise.py -d "${PROJECT_DIR}" -o "${OUTPUT_DIR}"
