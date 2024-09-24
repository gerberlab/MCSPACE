#!/bin/bash
set -e
source settings.sh

python cross_validation/helpers/generate_test_train_data.py -d "${PROJECT_DIR}" -o "${OUTPUT_DIR}"
