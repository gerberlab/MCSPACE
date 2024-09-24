#!/bin/bash

#SBATCH --partition=bwh_comppath
#SBATCH --job-name=cvGMMone
#SBATCH --cpus-per-task=40
#SBATCH --time=23:59:00
#SBATCH --output=log_cvGMMone.out
#SBATCH --error=errors_cvGMMone.out
#SBATCH --array=0-129

# Load a suitable module for python
module load miniconda3
conda activate mcfinal
cd ../../../scripts

bash cross_validation/run_cv_GMM_1dim.sh $SLURM_ARRAY_TASK_ID
