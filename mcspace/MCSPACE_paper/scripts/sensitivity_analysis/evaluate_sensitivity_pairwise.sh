#!/bin/bash
set -e
source settings.sh

python sensitivity_analysis/helpers/pairwise/evaluate_sensitivity_pairwise.py -d "${PROJECT_DIR}" -o "${OUTPUT_DIR}"
