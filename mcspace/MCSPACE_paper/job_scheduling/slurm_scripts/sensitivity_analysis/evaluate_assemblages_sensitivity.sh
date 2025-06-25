#!/bin/bash

#SBATCH --partition=bwh_comppath
#SBATCH --job-name=evalAR_sensitivity
#SBATCH --cpus-per-task=40
#SBATCH --time=23:59:00
#SBATCH --output=log_evalAR_sensitivity.out
#SBATCH --error=errors_evalAR_sensitivity.out

# Load a suitable module for python
module load miniconda3
conda activate mcspace
cd ../../../scripts

bash sensitivity_analysis/evaluate_sensitivity_assemblages.sh
