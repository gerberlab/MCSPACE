#!/bin/bash
set -e
source settings.sh

python sensitivity_analysis/helpers/assemblage_recovery/evaluate_sensitivity_assemblages.py -d "${PROJECT_DIR}" -o "${OUTPUT_DIR}"
