#!/bin/bash

#SBATCH --partition=bwh_comppath
#SBATCH --job-name=gen_cv
#SBATCH --cpus-per-task=40
#SBATCH --time=23:59:00
#SBATCH --output=log_gen_cv.out
#SBATCH --error=errors_gen_cv.out
# Load a suitable module for python

module load miniconda3
conda activate mcspace
cd ../../../scripts

bash cross_validation/generate_test_train_data.sh
