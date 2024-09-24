import os
os.environ["OMP_NUM_THREADS"] = "40" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "40" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "40" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "40" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "40" # export NUMEXPR_NUM_THREADS=6

import numpy as np
import torch
from mcspace.model import MCSPACE
from mcspace.trainer import train_model
from mcspace.data_utils import get_data, get_mouse_diet_perturbations_dataset
from mcspace.utils import get_device, pickle_load, pickle_save, get_summary_results, \
    MODEL_FILE, DATA_FILE, get_min_loss_path, get_posterior_summary_data, apply_taxonomy_threshold
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import mcspace.visualization as vis
from mcspace.dataset import DataSet
import pandas as pd


#! need to update data_utils loading...

def run_seed(outpathbase, datapath, seed):
    rootpath = Path("./")

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device()

    reads, num_otus, times, subjects, dataset = get_mouse_diet_perturbations_dataset(rootpath=datapath)
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

    # fig, ax = plt.subplots()
    # ax.plot(elbos)
    # ax.set_xlabel("Epoch")
    # ax.set_ylabel("ELBO loss")
    # plt.savefig(outpath / "elbo_loss.png", bbox_inches="tight")
    # plt.close()

def get_relative_abundances(data, times, subjects, taxonomy, multi_index=False):
    #! averaged over subjects
    reads = data['count_data']
    ntime = len(times)
    nsubj = len(subjects)
    notus = reads[times[0]][subjects[0]].shape[1]

    relabuns = np.zeros((notus, ntime, nsubj)) # also make into dataframe
    for i,t in enumerate(times):
        for j,s in enumerate(subjects):
            counts = reads[t][s].cpu().detach().clone().numpy()
            pra = counts/counts.sum(axis=1,keepdims=True)
            ras = np.mean(pra, axis=0)
            relabuns[:,i,j] = ras

    if multi_index is True:
        index = pd.MultiIndex.from_frame(taxonomy.reset_index())
    else:
        index = taxonomy.index 
    radf = pd.DataFrame(relabuns.mean(axis=2), index=index, columns=times)
    return radf


def get_best_run_summary(rootpath, runpath, seeds):
    respath = get_min_loss_path(runpath, seeds) 
    model = torch.load(respath / MODEL_FILE)
    data = pickle_load(respath / DATA_FILE)

    taxonomy = pickle_load(respath / "taxonomy.pkl")

    times = list(data['count_data'].keys())
    subjects = list(data['count_data'][10].keys())
    num_otus = data['count_data'][times[0]][subjects[0]].shape[1]
    num_times = len(times)
    num_subjects = len(subjects)
    
    taxonomy = apply_taxonomy_threshold(taxonomy)
    # TODO: do beforehand..
    name_updates = {'Otu10': {'Species': 'Faecalibaculum rodentium'},
    'Otu17': {'Genus': 'Roseburia'},
    'Otu6': {'Species': 'Eubacterium coprostanoligenes'},
    'Otu20': {'Species': 'Muribaculum gordoncarteri'},
    'Otu15': {'Genus': 'Eisenbergiella'},
    'Otu43': {'Family': 'Lachnospiraceae'}}

    taxonomy['Species'] = 'na'
    for oidx in name_updates.keys():
        replace = name_updates[oidx]
        key = list(replace.keys())[0]
        newname = replace[key]
        taxonomy.loc[oidx,key] = newname

    thetadf, betadf, pertsdf = get_posterior_summary_data(model, data, taxonomy, times, subjects)
    radf = get_relative_abundances(data, times, subjects, taxonomy)

    # save output
    outpath = rootpath / "results" / "analysis" / "Mouse"
    outpath.mkdir(exist_ok=True, parents=True)
    thetadf.to_csv(outpath / "assemblages.csv")
    betadf.to_csv(outpath / "assemblage_proportions.csv")
    pertsdf.to_csv(outpath / "perturbation_bayes_factors.csv")
    radf.to_csv(outpath / "relative_abundances.csv")


def main(rootdir, outdir):
    rootpath = Path(rootdir)
    datapath = rootpath / "datasets"
    basepath = Path(outdir) / "analysis" / "Mouse"
    outpathbase = basepath / "runs"
    outpathbase.mkdir(exist_ok=True, parents=True)

    seeds = np.arange(10)
    for seed in seeds:
        run_seed(outpathbase, datapath, seed)

    #! post process -> summary output
    print("outputting summary results for best run")
    get_best_run_summary(rootpath, outpathbase, seeds)
    print("***ALL DONE***")


if __name__ == "__main__":
    # 130 cases
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", dest='rootpath', help='root path')
    parser.add_argument("-o", dest='outpath', help='output path')
    args = parser.parse_args()
    main(args.rootpath, args.outpath)
