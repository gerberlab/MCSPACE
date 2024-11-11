#!/bin/bash
set -e
source settings.sh

python analysis/helpers/run_model_mouse_data.py -d "${PROJECT_DIR}" -o "${OUTPUT_DIR}"
