#!/bin/bash

#SBATCH --partition=bwh_comppath
#SBATCH --job-name=synGMM_human
#SBATCH --cpus-per-task=40
#SBATCH --time=23:59:00
#SBATCH --output=log_synGMM_human.out
#SBATCH --error=errors_synGMM_human.out
#SBATCH --array=0-199

# Load a suitable module for python
module load miniconda3
conda activate mcfinal
cd ../../../scripts

bash synthetic_benchmark/run_assemblage_recovery_GMM.sh Human $SLURM_ARRAY_TASK_ID
