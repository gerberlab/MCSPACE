#!/bin/bash

#SBATCH --partition=bwh_comppath
#SBATCH --job-name=synGMM_mouse
#SBATCH --cpus-per-task=40
#SBATCH --time=23:59:00
#SBATCH --output=log_synGMM_mouse.out
#SBATCH --error=errors_synGMM_mouse.out
#SBATCH --array=0-249

# Load a suitable module for python
module load miniconda3
conda activate mcspace
cd ../../../scripts

bash synthetic_benchmark/run_assemblage_recovery_GMM.sh Mouse $SLURM_ARRAY_TASK_ID
