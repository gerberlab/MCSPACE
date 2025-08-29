# MCSPACE sensitivity benchmarking analysis

These scripts perform the sensitivity benchmarking analysis required to reproduce **Figure TBD** in the paper. Model inference outputs are located in `MCSPACE_paper/output/sensitivity_analysis` and results for evaluation metrics are located in `MCSPACE_paper/results/sensitivity_analysis`. The Jupyter notebook to generate the resulting figures is located `MCSPACE_paper/figures/`

## Step 1: Generate semi-synthetic time-series data
First, generate synthetic datasets for sensitivity analysis using the command,
```bash
bash sensitivity_analysis/generate_synthetic_data_timeseries.sh
```

This generates the semi-synthetic datasets used for evaluating sensitivity benchmarking results. Resulting datasets are output into the folder `MCSPACE_paper/output/sensitivity_analysis/synthetic`

## Step 2: Run MCSPACE model inference on synthetic datasets
To run the MCSPACE model for inference on synthetic datasets, use the following command:
```bash
bash sensitivity_analysis/run_senstivity_MCSPACE.sh [run_all] [run_id]
```
### Running all cases sequentially
To execute all cases sequentially, run the command:
```bash
bash sensitivity_analysis/run_senstivity_MCSPACE.sh run_all
```
### Running specific cases
For running multiple cases in parallel (e.g. by using a job scheduler) or focusing on specific cases, you can provide a `run_id` parameter. To run a particular case, use the command:
```bash
bash sensitivity_analysis/run_senstivity_MCSPACE.sh [run_id]
```
* Options for `run_id`: 0 to 89

The script will run the MCSPACE model on the selected synthetic datasets and sensitivity parameter settings and save the inference results to the following directory: `MCSPACE_paper/output/sensitivity_analysis/assemblage_recovery/mcspace/`

## Step 3: Evaluate inference results for MCSPACE in recovering underlying assemblages in synthetic data
To evaluate benchmarking metrics on MCSPACE, after running inference, run the command:
```bash
bash sensitivity_analysis/evaluate_sensitivity_assemblages.sh
```
This computes the NMI, assemblage recovery error, and error in number of assemblages, and outputs the results in `MCSPACE_paper/results/sensitivity_analysis/assemblage_recovery/`

## Step 4: Evaluate pairwise inference results for MCSPACE
To compute the AUC for MCSPACE in recoverying pairwise associations, run the command:
```bash
bash sensitivity_analysis/evaluate_sensitivity_pairwise.sh
```
The results are output in `MCSPACE_paper/results/sensitivity_analysis/pairwise/mcspace_results`

