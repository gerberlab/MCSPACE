#!/bin/bash
set -e
source settings.sh

python synthetic_benchmark/helpers/pairwise/create_data_for_ecosim.py -d "${PROJECT_DIR}" -o "${OUTPUT_DIR}"
