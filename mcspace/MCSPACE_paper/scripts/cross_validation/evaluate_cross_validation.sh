#!/bin/bash
set -e
source settings.sh

python cross_validation/helpers/evaluate_results.py -d "${PROJECT_DIR}" -o "${OUTPUT_DIR}"
