#!/bin/bash
set -e
source settings.sh

python synthetic_benchmark/helpers/assemblage_recovery/evaluate_results.py -d "${PROJECT_DIR}" -o "${OUTPUT_DIR}"
