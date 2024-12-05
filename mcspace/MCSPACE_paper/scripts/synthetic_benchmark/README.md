# Synthetic benchmarking

These scripts perform the synthetic benchmarking analysis required to reproduce **Figure 3A-F** in the paper. Model inference outputs are located in `MCSPACE_paper/output/` and results for  evaluation metrics are located in `MCSPACE_paper/results/`. The Jupyter notebook to generate the resulting figures is located `MCSPACE_paper/figures/`

## Step 1: Generate semi-synthetic data
First, generate synthetic datasets for benchmarking using the command,
```bash
bash synthetic_benchmark/generate_synthetic_data.sh
```

This generates the semi-synthetic datasets used for evaluating benchmarking results. Resulting datasets are output into the folder `MCSPACE_paper/output/semisyn_data`

## Step 2: Run MCSPACE model inference on synthetic datasets
To run the MCSPACE model for inference on synthetic datasets, use the following command:
```bash
bash run_assemblage_recovery_MCSPACE.sh [run_all] Human [run_id]
```
### Running all cases sequentially
To execute all cases sequentially, run the command:
```bash
bash run_assemblage_recovery_MCSPACE.sh run_all
```
### Running specific cases
For running multiple cases in parallel (e.g. by using a job scheduler) or focusing on specific cases, you can provide a `run_id` parameter. To run a particular case, use the command:
```bash
bash run_assemblage_recovery_MCSPACE.sh Human [run_id]
```
* Options for `run_id`: 0 to 99

The script will run the MCSPACE model on the selected synthetic datasets and save the inference results to the following directory: `MCSPACE_paper/output/assemblage_recovery/mcspace/`

## Step 3: Run the GMM model on synthetic datasets
To run the Gaussian mixture model (GMM) for inference on synthetic datasets, use the following command:
```bash
bash run_assemblage_recovery_GMM.sh [run_all] Human [run_id]
```
### Running all cases sequentially
To execute all cases sequentially, run the command:
```bash
bash run_assemblage_recovery_GMM.sh run_all
```
### Running specific cases
For running multiple cases in parallel (e.g. by using a job scheduler) or focusing on specific cases, you can provide a `run_id` parameter. To run a particular case, use the command:
```bash
bash run_assemblage_recovery_GMM.sh Human [run_id]
```
* Options for `dataset`: 0 to 99

The script will run the GMM on the selected synthetic datasets and save the inference results to the following directory: `MCSPACE_paper/output/assemblage_recovery/gmm_basic_runs/`

## Step 4: Evaluate inference results for MCSPACE and GMM in recovering underlying assemblages in synthetic data
To evaluate benchmarking metrics on MCSPACE and GMM models, after running inference with each model, run the command:
```bash
bash evaluate_assemblage_recovery.sh
```
This computes the NMI, assemblage recovery error, and error in number of assemblages, and outputs the results in `MCSPACE_paper/results/assemblage_recovery/`

## Step 5: Evaluate pairwise inference results for MCSPACE
To compute the AUC for MCSPACE in recoverying pairwise associations, run the command:
```bash
bash evaluate_pairwise_MCSPACE.sh
```
The results are output in `MCSPACE_paper/results/pairwise/mcspace_results`

## Step 6: Run the Fisher test for pairwise analysis on synthetic datasets
To compute the AUC for the Fisher's exact test in recoverying pairwise associations, run the command:
```bash
bash run_pairwise_FISHER.sh
```
The results are output in `MCSPACE_paper/results/pairwise/fisher_results`

## Step 7: Run the SIM9 algorithm for pairwise analysis on synthetic datasets
To compute the AUC for the SIM9 algorithm in recoverying pairwise associations, perform the following steps:

### Step 7a: Generate binarized data for SIM9 algorithm
Generate csv files of binarized data for the SIM9 algorithm by running the command:
```bash
bash create_sim9_datafiles.sh
```

### Step 7b: Run SIM9 algorithm R script
The SIM9 algorithm is implemented as an R package. To run the SIM9 algorithm, run the `run_ecosim.R` file located in `MCSPACE_paper/scripts/synthetic_benchmark/helpers/pairwise/run_ecosim.R`.

This will generate result files, and outputs results in `MCSPACE_paper/results/assemblage_recovery/`.

**NOTE**: The ecosimR package required to run the SIM9 algorithm is no longer available on CRAN. It can be installed from source by downloading from `https://cran.r-project.org/src/contrib/Archive/EcoSimR/`.

### Step 7c: Evaluate SIM9 results and compute AUC metric
After performing step 7b, the results can be processed to compute the AUC by running the command:
```bash
bash evaluate_pairwise_SIM9.sh
```
