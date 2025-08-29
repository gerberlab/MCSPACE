# Navigation
This directory contains all scripts for reproducing the results in the main paper. Each subdirectory is a collection of scripts for running different parts of the analysis presented in the main paper.

All child scripts should be executed from this directory. For example, for generating synthetic datasets, you would run:
```bash
bash synthetic_benchmark/generate_synthetic_data.sh
```

Before running scripts, please ensure that the [MCSPACE](https://github.com/gerberlab/MCSPACE) package is installed. Refer to the main package github for instructions on installation.

For details on running a particular analysis, refer to the accompanying sub-documentation:
1. [Processing SAMPL-seq sequencing data](read_processing/README.md)
2. [Analysis of main datasets](analysis/README.md)
3. [Synthetic benchmarking](synthetic_benchmark/README.md)
4. [Cross validation](cross_validation/README.md)
5. [Sensitivity analysis](sensitivity_analysis/README.md)