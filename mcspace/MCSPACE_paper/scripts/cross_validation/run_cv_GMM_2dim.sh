#!/bin/bash
set -e
source settings.sh

if [[ "$1" == "run_all" ]]; then
	python cross_validation/helpers/run_gmm_two_dim.py -d "${PROJECT_DIR}" -o "${OUTPUT_DIR}" -run_all
else
	python cross_validation/helpers/run_gmm_two_dim.py -d "${PROJECT_DIR}" -o "${OUTPUT_DIR}" -idx "$1"
fi
