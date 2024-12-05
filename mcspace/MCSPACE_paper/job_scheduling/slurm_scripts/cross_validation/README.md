# Cross validation

These job submission scripts run the scripts in the `MCSPACE_paper/scripts/cross_validation` [directory](../../../scripts/cross_validation/README.md) to perform the cross-validated read prediction analysis.

## Step 1: Split data into test and train datasets 
```
sbatch create_cv_test_train_data.sh
```

## Step 2: Run MCSPACE model inference on training datasets
```
sbatch run_cv_MCSPACE.sh
```
This script deploys an array of 130 jobs, running MCSPACE on each training dataset sample.

## Step 3: Run Gaussian mixture model (GMM) inference on training datasets
```
sbatch run_cv_GMM.sh
```
This script deploys an array of 130 jobs, running GMM on each training dataset sample.

## Step 4: Run the one-dimensional directional GMM inference on training datasets
```
sbatch run_cv_GMM_onedim.sh
```
This script deploys an array of 130 jobs, running the one-dimensional directional GMM on each training dataset sample.

## Step 5: Run the two-dimensional directional GMM inference on training datasets
```
sbatch run_cv_GMM_twodim.sh
```
This script deploys an array of 130 jobs, running the two-dimensional directional GMM on each training dataset sample.

## Step 6: Evaluate predictive performance on test datasets
```
sbatch evaluate_cv.sh
```
