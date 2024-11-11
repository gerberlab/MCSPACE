#!/bin/bash

#SBATCH --partition=bwh_comppath
#SBATCH --job-name=synFISHER
#SBATCH --cpus-per-task=40
#SBATCH --time=23:59:00
#SBATCH --output=log_synFISHER.out
#SBATCH --error=errors_synFISHER.out

# Load a suitable module for python
module load miniconda3
conda activate mcspace
cd ../../../scripts

bash synthetic_benchmark/run_pairwise_FISHER.sh
