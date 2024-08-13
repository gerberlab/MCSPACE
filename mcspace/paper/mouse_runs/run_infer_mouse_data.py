import numpy as np
import torch
from mcspace.model import MCSPACE
from mcspace.trainer import train_model
from mcspace.data_utils import get_data, get_mouse_diet_perturbations_dataset
from mcspace.utils import get_device, pickle_load, pickle_save, get_summary_results, MODEL_FILE, DATA_FILE
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import mcspace.visualization as vis
from mcspace.dataset import DataSet
import pandas as pd


#! need to update data_utils loading...

def run_seed(outpathbase, seed):
    rootpath = Path("./")

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device()

    reads, num_otus, times, subjects, dataset = get_mouse_diet_perturbations_dataset(rootpath=rootpath)
    taxonomy = dataset.get_taxonomy()

    outpath = outpathbase / f"seed_{seed}"
    outpath.mkdir(exist_ok=True, parents=True)

    pickle_save(outpath / "taxonomy.pkl", taxonomy)

    # setup model
    data = get_data(reads, device)

    num_assemblages = 100
    perturbed_times = [0, 1, -1, 1, -1, 1, -1]
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
    num_epochs = 100 #5000
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


def main():
    rootpath = Path("./")
    basepath = rootpath / "paper" / "mouse_runs"

    seeds = np.arange(1) #0)

    outpath = basepath / "runs"
    outpath.mkdir(exist_ok=True, parents=True)

    for seed in seeds:
        run_seed(outpath, seed)
    

if __name__ == "__main__":
    main()
