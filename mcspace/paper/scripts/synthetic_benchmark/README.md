# Synthetic benchmarking

These scripts perform the synthetic benchmarking analysis. Outputs are located in `path/to/output/`. The Jupyter notebook to generate the resulting figure is located `path/to/notebook`.

# Step 1: Generate semi-synthetic data
```bash
bash synthetic_benchmark/generate_synthetic_data.sh
```

This generates and outputs synthetic datasets into the folder `path/to/synthetic data`.

# Step 2: Run MCSPACE model inference on synthetic datasets
```bash
bash run_assemblage_recovery_MCSPACE.sh
```

This runs the MCSPACE model on synthetic datasets and outputs model results to `path/to/mcspace/output`.

# Step 3: Run the GMM model on synthetic datasets
```bash
bash run_assemblage_recovery_GMM.sh
```

This runs a GMM model on synthetic datasets and outputs model results to `path/to/gmm/output`.

# Step 4: Run the Fisher test for pairwise analysis on synthetic datasets
```bash
bash run_pairwise_FISHER.sh
```

# Step 5: Run the SIM9 algorithm for pairwise analysis on synthetic datasets
```bash
bash run_pairwise_ECOSIM.sh
```

# Step 6: Evaluate inference results for MCSPACE and GMM in recovering underlying assemblages in synthetic data
```bash
bash evaluate_assemblage_recovery.sh
```

This outputs results in ...

# Step 7: Evaluate pairwise inference results for MCSPACE and pairwise methods
```bash
bash evaluate_pairwise.sh
```
