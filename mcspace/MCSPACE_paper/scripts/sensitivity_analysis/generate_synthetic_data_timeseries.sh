#!/bin/bash
set -e
source settings.sh

python sensitivity_analysis/helpers/synthetic/gen_semisyn_timeseries.py -d "${PROJECT_DIR}" -o "${OUTPUT_DIR}"
