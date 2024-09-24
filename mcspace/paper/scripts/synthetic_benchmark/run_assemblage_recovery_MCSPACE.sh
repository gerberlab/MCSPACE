#!/bin/bash
set -e
source settings.sh

if [[ "$1" == "run_all" ]]; then
	python synthetic_benchmark/helpers/assemblage_recovery/run_mcspace.py -d "${PROJECT_DIR}" -o "${OUTPUT_DIR}" -run_all
else
	python synthetic_benchmark/helpers/assemblage_recovery/run_mcspace.py -d "${PROJECT_DIR}" -o "${OUTPUT_DIR}" -dset "$1" -idx "$2"
fi
