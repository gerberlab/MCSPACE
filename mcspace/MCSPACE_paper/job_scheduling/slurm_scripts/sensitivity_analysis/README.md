# MCSPACE sensitivity analysis benchmarking

These job submission scripts run the scripts in the `MCSPACE_paper/scripts/sensitivity_analysis` [directory](../../../scripts/sensitivity_analysis/README.md) to perform the sensitivity analysis.

## Step 1: Generate semi-synthetic time-series data
```
sbatch generate_semisyn_timeseries_data.sh
```

## Step 2: Run MCSPACE model inference on synthetic datasets
The run model inference on the synthetic datasets, execute:
```
sbatch run_MCSPACE_sensitivity.sh
```
This will submit an array of 90 jobs, each running MCSPACE on a different semi-synthetic dataset and parameter condition.

## Step 3: Evaluate inference results for MCSPACE in recovering underlying assemblages in synthetic data and sensitivity to prior settings
```
sbatch evaluate_assemblages_sensitivity.sh
```

## Step 4: Evaluate pairwise inference results for MCSPACE
```
sbatch evaluate_pairwise_sensitivity.sh
```
