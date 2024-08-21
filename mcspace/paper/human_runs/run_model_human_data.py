import numpy as np
import torch
from mcspace.model import MCSPACE
from mcspace.trainer import train_model
from mcspace.data_utils import get_data, get_human_timeseries_dataset
from mcspace.utils import get_device, pickle_load, pickle_save, get_summary_results, MODEL_FILE, DATA_FILE
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import mcspace.visualization as vis
from mcspace.dataset import DataSet
import pandas as pd


def run_seed(outpathbase, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device()

    outpath = outpathbase / f"seed_{seed}"
    outpath.mkdir(exist_ok=True, parents=True)

    reads, num_otus, times, subjects, dataset = get_human_timeseries_dataset(min_abundance=0.005, min_reads=250)
    taxonomy = dataset.get_taxonomy()
    pickle_save(outpath / "taxonomy.pkl", taxonomy)
    pickle_save(outpath / "dataset.pkl", dataset)

    print(f"num otus = {num_otus}")
    print(f"times = {times}")
    print(f"subjects = {subjects}")

    # setup model
    data = get_data(reads, device)

    num_assemblages = 100
    perturbed_times = [0, 0, 0, 0, 0]
    perturbation_prior = 0.5/num_assemblages

    sparsity_prior = 0.5/num_assemblages

    num_reads = 0
    for t in times:
        for s in subjects:
            num_reads += reads[t][s].sum()
    sparsity_prior_power = 0.001*num_reads 

    process_var_prior = 0.01
    add_process_var=True

    model = MCSPACE(num_assemblages,
                num_otus,
                times,
                subjects,
                perturbed_times,
                perturbation_prior,
                sparsity_prior,
                sparsity_prior_power,
                process_var_prior,
                device,
                add_process_var,
                use_contamination=True,
                contamination_clusters=data['group_garbage_clusters']
                )
    model.to(device)

    # train model
    num_epochs = 5000
    elbos = train_model(model, data, num_epochs)

    #* save model and data
    torch.save(model, outpath / MODEL_FILE)
    pickle_save(outpath / DATA_FILE, data)

    fig, ax = plt.subplots()
    ax.plot(elbos)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ELBO loss")
    plt.savefig(outpath / "elbo_loss.png", bbox_inches="tight")
    plt.close()

    #* visualize results
    pert_bf, beta, theta, pi_garb, mean_loss = get_summary_results(model, data)
    otu_order, assemblage_order = vis.get_clustered_otu_assemblage_ordering(theta)
    labels = [f"Day {t}" for t in times]
    # taxonomy = dataset.get_taxonomy()
    vis.render_proportions_and_assemblages(beta, theta, taxonomy, otu_order, assemblage_order, ylabels=labels)
    plt.savefig(outpath / "assemblages_and_proportions.png", bbox_inches="tight")
    plt.close()


def main():
    rootpath = Path("./")
    basepath = rootpath / "paper" / "human_dataset_runs"
    outpathbase = basepath / "runs_FF"
    outpathbase.mkdir(exist_ok=True, parents=True)

    seeds = np.arange(10)
    for seed in seeds: #[42]:
        run_seed(outpathbase, seed)

    
if __name__ == "__main__":
    main()
