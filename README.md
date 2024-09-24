# MCSPACE
Sparse Bayesian mixture model for learning community subtypes and perturbation effects from MaPS-seq experimental data.

<p align="center">
<img src="/media/mcspace_overview.svg" />
</p>


## Install from source
First install the MCSPACE package:
```
git clone https://github.com/gerberlab/MCSPACE.git
pip install MCSPACE/.
```

Then install [pytorch](https://pytorch.org/) from pip

## Usage
We provide tutorials for running the MCSPACE model on a single MaPS-seq data sample [here](https://github.com/gerberlab/MCSPACE/blob/main/mcspace/tutorials/tutorial_single_sample.ipynb), as well as running inference on a perturbation study [here](https://github.com/gerberlab/MCSPACE/blob/main/mcspace/tutorials/tutorial_perturbation_example.ipynb). These show how to use MCSPACE to load data, run the model, and use built in visualization to interpret results.
