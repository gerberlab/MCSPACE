#!/bin/bash

#SBATCH --partition=bwh_comppath
#SBATCH --job-name=synMC_sensitivity
#SBATCH --cpus-per-task=40
#SBATCH --time=23:59:00
#SBATCH --output=log_synMC_sensitivity.out
#SBATCH --error=errors_synMC_sensitivity.out
#SBATCH --array=0-89

# Load a suitable module for python
module load miniconda3
conda activate mcspace
cd ../../../scripts

bash sensitivity_analysis/run_senstivity_MCSPACE.sh $SLURM_ARRAY_TASK_ID
