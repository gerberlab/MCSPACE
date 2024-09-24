#!/bin/bash

#SBATCH --partition=bwh_comppath
#SBATCH --job-name=evalAR
#SBATCH --cpus-per-task=40
#SBATCH --time=23:59:00
#SBATCH --output=log_evalAR.out
#SBATCH --error=errors_evalAR.out

# Load a suitable module for python
module load miniconda3
conda activate mcfinal
cd ../../../scripts

bash synthetic_benchmark/evaluate_assemblage_recovery.sh
