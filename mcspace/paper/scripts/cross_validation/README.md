# Cross validation

These scripts perform the cross-validated read prediction analysis to reproduce the results of **Figure 4** in the paper. Model inference outputs are located in `paper/output/cross_validation` and results for evaluation metrics are located in `paper/results/cross_validation`. The Jupyter notebook to generate the resulting figures is located `paper/figures/`

## Step 1: Split data into test and train datasets 
First, we generate 5-fold train-test splits of the human and mouse datasets, using the command,
```bash
bash cross_validation/generate_test_train_data.sh
```
This splits each dataset sample into (80/20) train-test splits and performs a 50% down-sampling of reads in the test datasets. The resulting datasets are output into the folder: `paper/output/cross_validation/holdout_data/`.

## Step 2: Run MCSPACE model inference on training datasets
To run the MCSPACE model on training datasets, execute the command:
```bash
bash run_cv_MCSPACE.sh [run_all] [run_id]
```
### Running all cases sequentially
To execute all cases sequentially, run the command:
```bash
bash run_cv_MCSPACE.sh run_all
```
### Running specific cases
For running multiple cases in parallel (e.g. by using a job scheduler) or focusing on specific cases, you can provide the `run_id` parameter. To run a particular case, use the command:
```bash
bash run_cv_MCSPACE.sh [run_id]
```
* Options for `run_id`: 0 to 129

The model inference results are output into the folder: `paper/output/cross_validation/mcspace_runs`

## Step 3: Run Gaussian mixture model (GMM) inference on training datasets
To run the GMM on training datasets, execute the command:
```bash
bash run_cv_GMM.sh [run_all] [run_id]
```
### Running all cases sequentially
To execute all cases sequentially, run the command:
```bash
bash run_cv_GMM.sh run_all
```
### Running specific cases
For running multiple cases in parallel (e.g. by using a job scheduler) or focusing on specific cases, you can provide the `run_id` parameter. To run a particular case, use the command:
```bash
bash run_cv_GMM.sh [run_id]
```
* Options for `run_id`: 0 to 129

The model inference results are output into the folder: `paper/output/cross_validation/gmm_basic`

## Step 4: Run the one-dimensional directional GMM inference on training datasets
To run the one-dimensional directional GMM model on training datasets, execute the command:
```bash
bash run_cv_GMM_1dim.sh [run_all] [run_id]
```
### Running all cases sequentially
To execute all cases sequentially, run the command:
```bash
bash run_cv_GMM_1dim.sh run_all
```
### Running specific cases
For running multiple cases in parallel (e.g. by using a job scheduler) or focusing on specific cases, you can provide the `run_id` parameter. To run a particular case, use the command:
```bash
bash run_cv_GMM_1dim.sh [run_id]
```
* Options for `run_id`: 0 to 129

The model inference results are output into the folder: `paper/output/cross_validation/gmm_one_dim`

## Step 5: Run the two-dimensional directional GMM inference on training datasets

To run the two-dimensional directional GMM model on training datasets, execute the command:
```bash
bash run_cv_GMM_2dim.sh [run_all] [run_id]
```
### Running all cases sequentially
To execute all cases sequentially, run the command:
```bash
bash run_cv_GMM_2dim.sh run_all
```
### Running specific cases
For running multiple cases in parallel (e.g. by using a job scheduler) or focusing on specific cases, you can provide the `run_id` parameter. To run a particular case, use the command:
```bash
bash run_cv_GMM_2dim.sh [run_id]
```
* Options for `run_id`: 0 to 129

The model inference results are output into the folder: `paper/output/cross_validation/gmm_two_dim`

## Step 6: Evaluate predictive performance on test datasets
To evaluate model prediction performance on heldout test datasets, execute the command
```bash
bash evaluate_cross_validation.sh
```

This script calculates the cosine distance error in predicting heldout reads for each particle in the test datasets and outputs results to the folder: `paper/results/cross_validation`
