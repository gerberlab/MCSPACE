#!/bin/bash

#SBATCH --partition=bwh_comppath
#SBATCH --job-name=pwMCSPACE
#SBATCH --cpus-per-task=40
#SBATCH --time=23:59:00
#SBATCH --output=log_pwMCSPACE.out
#SBATCH --error=errors_pwMCSPACE.out

# Load a suitable module for python
module load miniconda3
conda activate mcspace
cd ../../../scripts

bash synthetic_benchmark/evaluate_pairwise_MCSPACE.sh
