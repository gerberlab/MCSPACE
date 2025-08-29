#!/bin/bash

#SBATCH --partition=bwh_comppath
#SBATCH --job-name=gen_syn
#SBATCH --cpus-per-task=40
#SBATCH --time=23:59:00
#SBATCH --output=log_gen_syn.out
#SBATCH --error=errors_gen_syn.out
# Load a suitable module for python

module load miniconda3
conda activate mcspace
cd ../../../scripts

bash sensitivity_analysis/generate_synthetic_data_timeseries.sh
