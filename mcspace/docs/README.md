# MCSPACE user's manual

## Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Required input data and formatting](#required-input-data-and-formatting)
- [Preparing data for model inference](#preparing-data-for-model-inference)
- [Running inference](#running-inference)
- [Visualizing results](#visualizing-results)

## Introduction
The MCSPACE software package identifies spatially co-localized microbial groups, termed *assemblages*, from sequencing data and tracks changes in their proportions over time or in response to perturbations.

MCSPACE is implemented as a python library and as a command line interface (CLI). The library can be imported using the command: `import mcspace`, and the CLI is accessed using the command `mcspace`. The MCSPACE pipeline consists of three main steps: data preparation, inference, and result visualization. Below we describe the key functions that perform each of these steps. Accompanying [tutorials](../tutorials/) also provide examples for each step.

## Installation
### Install the MCSPACE package from pip via source:
```
git clone https://github.com/gerberlab/MCSPACE.git
pip install MCSPACE/.
```

### Install [pytorch](https://pytorch.org/) from pip

#### Linux or Windows (with NVIDIA GPU and CUDA 11.8)
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Linux or Windows (CPU only)
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### MacOS (CUDA not supported)
```
pip3 install torch torchvision torchaudio
```

## Required input data and formatting
The MCSPACE software requires three files: a **counts** file,  a **taxonomy** file, and a **perturbations** file. Each must be formatted as described below. See also the [tutorial](../tutorials/data_preprocessing.ipynb) for examples of each file.

### Counts:
This is a csv file that gives particle sequence count data for each OTU for all samples in the study in a long format.

The following columns are **required** to be in this file:
- `Particle`: This column contains the unique particle ID for each particle in the study.
- `OTU`: This column gives the Otu# to which the counts correspond to for each row.
- `Count`: Each row in this column gives the number of sequencing counts corresponding to a given OTU in a given particle, in a given sample.
- `Time`: Each row in this column gives the timepoint to which the counts correspond to.
- `Subject`: Each row in this column gives the subject to which the counts correspond to.

### Taxonomy:
This is a csv file that contains taxonomic information for OTUs in the study. It requires the following columns: `Otu, Domain, Phylum, Class, Order, Family, Genus, Species`. A value of `na` in an entry indicates that the OTU is not resolved to the corresponding taxonomic level. When visualizing results the software automatically displays each OTU to its lowest resolved taxonomic level.

### Perturbations:
This is a csv file with two columns, containing information on which time points correspond to experimental perturbations. The csv file **requires** two columns:
- `Time`: Each row listing each timepoint in the study
- `Perturbed`: Each row must contain either a 0 or 1, with 0 indicating no perturbation on the corresponding timepoint and a 1 indicating the timepoint does correspond to a perturbation


## Preparing data for model inference
Data is prepared for model inference using the `parse` function as described below. See the [tutorial](../tutorials/data_preprocessing.ipynb) for an example on how to use this function.

### `parse`:

**Required arguments**:
- `counts_data`: The first argument of the parse function takes the filepath for the counts file, as described above.
- `taxonomy`: The second argument of the parse function takes the taxonomy filepath.
- `perturbation_info`: The third argument of the parse function takes the filepath for the file containing perturbation information for the timepoints in the study.

**Optional keyword arguments**:
- `subjects_remove`: This argument takes in a list of subjects to be removed from the study. Default value is `None`.
- `times_remove`: This arugment takes in a list of timepoints to be removed from the study. Default value is `None`.
- `otus_remove`: This argument takes in a list of OTUs to be removed from the study. Default value is `None`.
- `num_consistent_subjects`: This is the number of subjects that must contain each OTU above the provided `min_abundance` for it to be included. Default value is 1.
- `min_abundance`: This is the minimum relative abundance an OTU must have on any timepoint for it to be included. The default value is 0.005.
- `min_reads`: This is the minimum number of reads a particle must contain for it to be included. The default value is 250.
- `max_reads`: This is the maximum number of reads a particle can contain for it to be included. The default value is 10000.

**Returns**:
- `processed_data`: A dictionary of objects used in the MCSPACE inference step.

Example: `processed_data = parse(counts.csv,taxonomy.csv,perturbations.csv)`


## Running inference
Model inference is perfomed using the `run_inference` function as described below. See the [tutorial](../tutorials/running_inference.ipynb) for an example on performing the running model inference step.

### `run_inference`:
**Required arguments**:
- `data`: The first argument takes the preprocessed data which is the resulting output from the **parse** function.
- `outpath`: The second argument takes a path to the directory to which the results of inference are to be saved.

**Optional keyword arguments**:
- `n_seeds`: This argument corresponds to how many resets are to be used for model inference. The model is then run `n_seeds` number of times and the best model is selected as the one with the lowest ELBO loss. The default value is 10.
- `n_epochs`: This is the number of training epochs to use in each reset. Default value is 20000.
- `learning_rate`: Learning rate to be used with the ADAM optimizer. Default  value is 5e-3.
- `num_assemblages`: Maximum possible assemblages the model can learn. Default value is 100.
- `sparsity_prior`: Prior probability of an assemblage being present. The default value is None, which sets the value to 0.5/`num_assemblages`.
- `sparsity_power`: Power to which we raise the sparsity prior to scale with the dataset size. Default value is `None` which sets the value to 0.5% of the total number of reads in the dataset.
- `anneal_prior`: Specifies whether to anneal the strength of the sparsity prior during training. Default value is True.
- `process_variance_prior`: Prior location of the process variance prior. Default value is `0.01`.
- `perturbation_prior`: Prior probability of a perturbation effect. Default value is None, which sets the value to 0.5/`num_assemblages`.
- `use_contamination`: Whether to use the contamination cluster in the model. Default value is True.
- `use_sparsity`: Specifies whether to sparsify the number of assemblages in the model. Default value is True.
- `use_kmeans_init`: Specifies whether to use a kmeans initialization for assemblage parameters. Default value is True.
- `device`: Specifies whether to use the CPU or GPU for model inference. By default, the software automatically detects and utilizes the GPU if available.

Example: `run_inference(processed_data, "./results/")`

## Visualizing results
Methods for visualizing results include: `render_assemblages`, `render_assemblage_proportions`, and `export_association_networks_to_cytoscape`, each described below. See the [tutorial](../tutorials/visualizating_results.ipynb) for examples on using the visualization methods.

### Visualizing assemblages:
We visualize assemblages with the function `render_assemblages` as described below:

### `render_assemblages`:
**Required arguments**
- `results`: The first required argument is the results given from performing model inference.
- `outfile`: Filename to which to save the resulting figure.

**Optional keyword arguments**:
- `otu_threshold`: Filtering threshold below which to exclude OTUs in assemblages. Default value is 0.05.
- `treefile`: Filename of phylogenetic tree for OTUs in study, in Newick format. This file is not required but can be provided to visualize assemblages on the tree. If not provided, assemblages are plotted without the tree.
- `fontsize`: Fontsize for text in figure. Default value is 6.
- `legend`: Specifies whether to include figure legend. Default value is True.

**Returns**:
- `ax_tree`: Axis object for phylogenetic tree.
- `ax_theta`: Axis object for assemblage heatmap.
- `ax_cbar`: Axis object for legend colorbar.


### Visualizing assemblage proportions
We visualize assemblage proportions using the `render_assemblage_proportions` function as decribed below:

### `render_assemblage_proportions`
**Required parameters**:
- `results`: The first required argument is the results given from performing model inference.
- `outfile`: Filename to which to save the resulting figure.

**Optional keyword arguments**:
- `average_subjects`: Specifies whether to average assemblage proportions over subjects and display only the subject averaged values for each timepoint. Default value is False.
- `annotate_bayes_factors`: Specifies whether to display perturbation Bayes factors, if perturbations are included in the study. Default value is False.
- `logscale`: Specifies whether to use a logscale for assemblage proportions. Default value is True. Providing the value False will plot proportions on a linear scale.
- `beta_vmin`: Minimum value for heatmap. If using logscale, corresponds to power of 10. Default value is -3, corresponding to 0.001.
- fontsize=6,
- `fontsize`: Fontsize for text in figure. Default value is 6.
- `legend`: Specifies whether to include figure legend. Default value is True.

**Returns**:
- `ax`: Array of axis objects for heatmap for each timepoint if plotting all subjects. If averaging subjects, the axis consists of just a single object for the heatmap.
- `ax_cbar`: Axis object for legend colorbar.
- `ax_bf`: Axis object for Bayes Factor legend.


### Exporting association networks for cytoscape
The function `export_association_networks_to_cytoscape` exports association networks over time, for a specified taxon to an xml file which can be interactively explored in cytoscape. The function is described below:

### `export_association_networks_to_cytoscape`
**Required arguments**:
- `Otu_ID`: The first required argument takes the Otu name for hub taxon for which to export association networks. 
- `results`: The second required argument is the results given from performing model inference.
- `outfile`: The third required argument takes the filename to which to export the association networks to.

**Optional keyword arguments**:
- `ra_threshold`: Relative abundance threshold for which taxa to include in network. Taxa with a relative abundance below the specified threshold at all timepoints in the study will be excluded. Default value is 0.01.
- `edge_threshold`: Association score threshold for excluding taxa from the network. Taxa with an association score less than the specified value on all timepoints will be excluded. Default value is 0.01.

For example, to export networks for **Otu2**, to the file **otu2_associations.xml** we execute the following command:
```
export_association_networks_to_cytoscape("Otu2",
                                         results,
                                         outpath/"otu2_associations.xml")
```
