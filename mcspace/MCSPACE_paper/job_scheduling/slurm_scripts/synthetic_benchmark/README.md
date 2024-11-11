# Synthetic benchmarking

These job submission scripts run the scripts in the `paper/scripts/synthetic_benchmark` [directory](../../../scripts/synthetic_benchmark/README.md) to perform the synthetic benchmarking analysis.

## Step 1: Generate semi-synthetic data
```
sbatch generate_semisyn_data.sh
```

## Step 2: Run MCSPACE model inference on synthetic datasets
The model inference is split into two parts: one for running the human synthetic datasets and one for running the mouse synthetic datasets.

To run the Human datasets, execute:
```
sbatch run_MCSPACE_Human.sh
```
This will submit an array of 200 jobs, each running MCSPACE on a different Human synthetic dataset condition.

To run the Mouse datasets, execute:
```
sbatch run_MCSPACE_Mouse.sh
```
This will submit an array of 250 jobs, each running MCSPACE on a different Mouse synthetic dataset condition.

## Step 3: Run the GMM model on synthetic datasets
The GMM inference is split into two parts: one for running the human synthetic datasets and one for running the mouse synthetic datasets.

To run the Human datasets, execute:
```
sbatch run_GMM_Human.sh
```
This will submit an array of 200 jobs, each running the GMM on a different Human synthetic dataset condition.

To run the Mouse datasets, execute:
```
sbatch run_GMM_Mouse.sh
```
This will submit an array of 250 jobs, each running the GMM on a different Mouse synthetic dataset condition.

## Step 4: Evaluate inference results for MCSPACE and GMM in recovering underlying assemblages in synthetic data
```
sbatch eval_assemblage_recovery.sh
```

## Step 5: Evaluate pairwise inference results for MCSPACE
```
sbatch eval_pairwise_MCSPACE.sh
```

## Step 6: Run the Fisher test for pairwise analysis on synthetic datasets
```
sbatch run_FISHER.sh
```

## Step 7: Run the SIM9 algorithm for pairwise analysis on synthetic datasets
### TO DO...
