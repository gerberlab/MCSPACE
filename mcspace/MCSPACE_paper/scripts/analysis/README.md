# Analysis of main datasets

These scripts perform MCSPACE model inference on the human and mouse datasets (located in `MCSPACE_paper/datasets/`) analyzed in the paper.

## Run MCSPACE on Human time series dataset
To run MCSPACE on the Human time series dataset analyzed in the paper, execute the command:
```bash
bash analysis/run_MCSPACE_human_data.sh
```
This runs the MCSPACE model on the Human dataset with 10 different initial seeds. Each model inference run is saved in the folder: `MCSPACE_paper/output/analysis/Human/runs`. The best run, corresponding to the run with the best ELBO loss, is then selected and posterior summary results for learned assemblages and assemblage proportions of the best run are output into the folder: `MCSPACE_paper/results/analysis/Human`.

## Run MCSPACE on human with inulin supplementation dataset
To run MCSPACE on the human study with inulin supplementation dataset analyzed in the paper, execute the command:
```bash
bash analysis/run_MCSPACE_human_inulin_data.sh
```
This runs the MCSPACE model on the dataset with 10 different initial seeds. Each model inference run is saved in the folder: `MCSPACE_paper/output/analysis/human_inulin/runs`. The best run, corresponding to the run with the best ELBO loss, is then selected and posterior summary results for learned assemblages and assemblage proportions of the best run are output into the folder: `MCSPACE_paper/results/analysis/human_inulin`.

## Run MCSPACE on Mouse dataset
To run MCSPACE on the Mouse dataset analyzed in the paper, execute the command:
```bash
bash analysis/run_MCSPACE_mouse_data.sh
```
This runs the MCSPACE model on the Mouse dataset with 10 different initial seeds. Each model inference run is saved in the folder: `MCSPACE_paper/output/analysis/Mouse/runs`. The best run, corresponding to the run with the best ELBO loss, is then selected and posterior summary results for learned assemblages, assemblage proportions, and perturbation Bayes factors of the best run are output into the folder: `paper/results/analysis/Mouse`.
