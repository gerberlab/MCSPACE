#!/bin/bash
set -e
source settings.sh

python synthetic_benchmark/helpers/synthetic/gen_semisyn_data.py -d "${PROJECT_DIR}" -o "${OUTPUT_DIR}"
