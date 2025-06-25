#!/bin/bash
set -e
source settings.sh

if [[ "$1" == "run_all" ]]; then
	python sensitivity_analysis/helpers/assemblage_recovery/run_MCSPACE_sensitivity.py -d "${PROJECT_DIR}" -o "${OUTPUT_DIR}" -run_all
else
	python sensitivity_analysis/helpers/assemblage_recovery/run_MCSPACE_sensitivity.py -d "${PROJECT_DIR}" -o "${OUTPUT_DIR}" -idx "$1"
fi
