# Job scheduling

This directory contains all the scripts used to run the main paper analysis on the ERISTwo compute cluster using the SLURM workload manager. Each batch job distribution script calls one of the main analysis scripts in the `paper/scripts/` directory. 

Each script should be called from the directory it is in, for example, for generating synthetic datasets, you would run
```
sbatch generate_semisyn_data.sh
```
from within the `job_scheduling/slurm_scripts/benchmarking/` directory.

Before running scripts, please ensure that the [MCSPACE](https://github.com/gerberlab/MCSPACE) package is installed. Refer to the main package github for instructions on installation.

For details on running a particular analysis, refer to the accompanying sub-documentation:
1. [Analysis of main datasets](analysis/README.md)
2. [Synthetic benchmarking](synthetic_benchmark/README.md)
3. [Cross validation](cross_validation/README.md)
