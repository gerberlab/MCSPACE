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
from mcspace.data_utils import get_data, get_human_timeseries_dataset
from mcspace.utils import get_device, pickle_load, pickle_save, get_summary_results, \
    MODEL_FILE, DATA_FILE, get_min_loss_path, get_posterior_summary_data
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import mcspace.visualization as vis
from mcspace.dataset import DataSet
import pandas as pd


def run_seed(outpathbase, datapath, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device()

    outpath = outpathbase / f"seed_{seed}"
    outpath.mkdir(exist_ok=True, parents=True)

    reads, num_otus, times, subjects, dataset = get_human_timeseries_dataset(rootpath=datapath)
    taxonomy = dataset.get_taxonomy()
    pickle_save(outpath / "taxonomy.pkl", taxonomy)
    # pickle_save(outpath / "dataset.pkl", dataset)

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

    # fig, ax = plt.subplots()
    # ax.plot(elbos)
    # ax.set_xlabel("Epoch")
    # ax.set_ylabel("ELBO loss")
    # plt.savefig(outpath / "elbo_loss.png", bbox_inches="tight")
    # plt.close()


def get_bulk_relative_abundances(reads, times, subjects, taxonomy):
    num_otus = taxonomy.shape[0]
    multiind = pd.MultiIndex.from_frame(taxonomy)
    ntime = len(times)
    
    radfs = {}
    for s in subjects:
        ra = np.zeros((ntime, num_otus))
        for i,t in enumerate(times):
            counts = reads[t][s].cpu().detach().clone().numpy()
            rabun = counts/counts.sum(axis=1, keepdims=True)
            bulk_rabun = np.mean(rabun, axis=0)
            ra[i,:] = bulk_rabun
        df = pd.DataFrame(data=ra.T, index=multiind, columns=times)
        radfs[s] = df
    return radfs


def get_best_run_summary(rootpath, runpath, seeds):
    respath = get_min_loss_path(runpath, seeds) 
    model = torch.load(respath / MODEL_FILE)
    data = pickle_load(respath / DATA_FILE)

    taxonomy_temp = pickle_load(respath / "taxonomy.pkl")

    times = list(data['count_data'].keys())
    subjects = list(data['count_data'][1].keys())
    num_otus = data['count_data'][times[0]][subjects[0]].shape[1]
    num_times = len(times)
    num_subjects = len(subjects)
    taxfile =  rootpath / "datasets" / "human_experiments" / "gappa_taxonomy" / "human_taxonomy.csv"
    finaltax = pd.read_csv(taxfile, index_col=0)
    taxlist = list(taxonomy_temp.index)
    taxonomy = finaltax.loc[taxlist,:]
    thetadf, betadf, pertsdf = get_posterior_summary_data(model, data, taxonomy, times, subjects)
    reads = data['count_data'] #.cpu().detach().clone().numpy()
    bulktemp = get_bulk_relative_abundances(reads, times, subjects, taxonomy.reset_index())
    bulk = bulktemp['H11'].reset_index()[['Otu'] + times].set_index('Otu')

    # save output
    outpath = rootpath / "results" / "analysis" / "Human"
    outpath.mkdir(exist_ok=True, parents=True)
    thetadf.to_csv(outpath / "assemblages.csv")
    betadf.to_csv(outpath / "assemblage_proportions.csv")
    bulk.to_csv(outpath / "relative_abundances.csv")
    taxonomy.to_csv(outpath / "taxonomy.csv")


def main(rootdir, outdir):
    rootpath = Path(rootdir)
    datapath = rootpath / "datasets"
    basepath = Path(outdir) / "analysis" / "Human"
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
