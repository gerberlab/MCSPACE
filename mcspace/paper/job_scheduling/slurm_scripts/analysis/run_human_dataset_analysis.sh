#!/bin/bash

#SBATCH --partition=bwh_comppath
#SBATCH --job-name=rHuman
#SBATCH --cpus-per-task=40
#SBATCH --time=23:59:00
#SBATCH --output=log_rHuman.example
#SBATCH --error=errors_rHuman.example
# Load a suitable module for python

module load miniconda3
conda activate mcspace
cd ../../../scripts

bash analysis/run_MCSPACE_human_data.sh
