# Navigation

Each subdirectory is a collection of scripts for running different parts of the analysis presented in the main paper.

All child scripts should be executed from this directory. For example, for generating synthetic datasets, you would run:
```bash
bash synthetic_benchmark/generate_synthetic_data.sh
```

For details on running a particular analysis, refer to the accompanying sub-documentation:
1. [Analysis of main datasets](analysis/README.md)
2. [Synthetic benchmarking](synthetic_benchmark/README.md)
3. [Cross validation](cross_validation/README.md)
