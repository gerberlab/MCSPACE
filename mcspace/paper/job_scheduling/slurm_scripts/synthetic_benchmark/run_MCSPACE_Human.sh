#!/bin/bash

#SBATCH --partition=bwh_comppath
#SBATCH --job-name=synMC_human
#SBATCH --cpus-per-task=40
#SBATCH --time=23:59:00
#SBATCH --output=log_synMC_human.out
#SBATCH --error=errors_synMC_human.out
#SBATCH --array=0-199

# Load a suitable module for python
module load miniconda3
conda activate mcspace
cd ../../../scripts

bash synthetic_benchmark/run_assemblage_recovery_MCSPACE.sh Human $SLURM_ARRAY_TASK_ID
