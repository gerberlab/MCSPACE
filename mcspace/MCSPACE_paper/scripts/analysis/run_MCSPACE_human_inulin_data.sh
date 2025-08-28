#!/bin/bash
set -e
source settings.sh

python analysis/helpers/run_model_human_inulin.py -d "${PROJECT_DIR}" -o "${OUTPUT_DIR}"
