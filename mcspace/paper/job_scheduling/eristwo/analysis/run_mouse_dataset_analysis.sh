#!/bin/bash

#SBATCH --partition=bwh_comppath
#SBATCH --job-name=rMouse
#SBATCH --cpus-per-task=40
#SBATCH --time=23:59:00
#SBATCH --output=log_rMouse.example
#SBATCH --error=errors_rMouse.example
# Load a suitable module for python

module load miniconda3
conda activate mcfinal
cd ../../../scripts

bash analysis/run_MCSPACE_mouse_data.sh
