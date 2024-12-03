import numpy as np
import pandas as pd
import torch
from mcspace.model import MCSPACE
from mcspace.trainer import train_model
from mcspace.utils import get_device, pickle_load, pickle_save, \
    MODEL_FILE, DATA_FILE, get_min_loss_path, get_posterior_summary_data
from pathlib import Path


def get_relative_abundances(data, times, subjects, taxonomy):
    # ! averaged over subjects
    reads = data['count_data']
    ntime = len(times)
    nsubj = len(subjects)
    notus = reads[times[0]][subjects[0]].shape[1]

    relabuns = np.zeros((notus, ntime, nsubj))  # also make into dataframe
    for i, t in enumerate(times):
        for j, s in enumerate(subjects):
            counts = reads[t][s].cpu().detach().clone().numpy()
            pra = counts / counts.sum(axis=1, keepdims=True)
            ras = np.mean(pra, axis=0)
            relabuns[:, i, j] = ras

    index = taxonomy.index
    radf = pd.DataFrame(relabuns.mean(axis=2), index=index, columns=times)
    return radf


def save_best_model_posterior_summary(runs_outpath, seeds, outpath):
    respath = get_min_loss_path(runs_outpath, seeds)
    model = torch.load(respath / MODEL_FILE)
    data = pickle_load(respath / DATA_FILE)
    taxonomy = pickle_load(respath / "taxonomy.pkl")

    # save best model
    best_outpath = outpath / "best_model"
    best_outpath.mkdir(exist_ok=True, parents=True)
    torch.save(model, best_outpath / MODEL_FILE)
    pickle_save(best_outpath / DATA_FILE, data)
    pickle_save(best_outpath / "taxonomy.pkl", taxonomy)

    # get posterior summary and save
    times = list(data['count_data'].keys())
    subjects = list(data['count_data'][times[0]].keys())
    thetadf, betadf, pertsdf = get_posterior_summary_data(model, data, taxonomy, times, subjects)
    radf = get_relative_abundances(data, times, subjects, taxonomy)

    # save output
    thetadf.to_csv(outpath / "assemblages.csv")
    betadf.to_csv(outpath / "assemblage_proportions.csv")
    if pertsdf is not None:
        pertsdf.to_csv(outpath / "perturbation_bayes_factors.csv")
    radf.to_csv(outpath / "relative_abundances.csv")
    # summary dict
    results = {'assemblages': thetadf,
               'assemblage_proportions': betadf,
               'perturbation_bayes_factors': pertsdf,
               'relative_abundances': radf}
    pickle_save(outpath / "results.pkl", results)


def run_inference_seed(data,
                      runs_outpath,
                      seed,
                      n_epochs,
                      learning_rate,
                      num_assemblages,
                      sparsity_prior,
                      sparsity_power,
                      anneal_prior,
                      process_variance_prior,
                      perturbation_prior,
                      use_contamination,
                      use_sparsity,
                      use_kmeans_init,
                      device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"running seed {seed}...")
    outpath = runs_outpath / f"seed_{seed}"
    outpath.mkdir(exist_ok=True, parents=True)

    inference_data = data['inference_data']
    dataset = data['dataset']
    perturbed_times = data['perturbations']
    taxonomy = data['taxonomy']
    num_otus = len(dataset.otu_index)
    times = dataset.times
    subjects = dataset.subjects
    if len(times) > 2:
        is_time_series = True
    else:
        is_time_series = False

    model = MCSPACE(num_assemblages,
                num_otus,
                times,
                subjects,
                perturbed_times,
                perturbation_prior,
                sparsity_prior,
                sparsity_power,
                process_variance_prior,
                device,
                add_process_variance=is_time_series,
                use_sparse_weights=use_sparsity,
                use_contamination=use_contamination,
                contamination_clusters=inference_data['group_garbage_clusters'],
                lr=learning_rate
                )
    if use_kmeans_init is True:
        model.kmeans_init(inference_data, seed)
    model.to(device)

    # train model
    num_epochs = n_epochs
    elbos = train_model(model, inference_data, num_epochs, anneal_prior=anneal_prior)

    #* save model and data
    torch.save(model, outpath / MODEL_FILE)
    pickle_save(outpath / DATA_FILE, inference_data)
    pickle_save(outpath / "taxonomy.pkl", taxonomy)


def run_inference(data,
                  outdir,
                  n_seeds=10,
                  n_epochs=20000,
                  learning_rate=5e-3,
                  num_assemblages=100,
                  sparsity_prior=None, # set to 0.5/K
                  sparsity_power=None, # get from data in function
                  anneal_prior=True,
                  process_variance_prior=0.01,
                  perturbation_prior=None, # set to 0.5/K
                  use_contamination=True,
                  use_sparsity=True,
                  use_kmeans_init=True,
                  device=get_device()):

    outpath = Path(outdir)
    runs_outpath = outpath / "runs"
    outpath.mkdir(exist_ok=True, parents=True)
    runs_outpath.mkdir(exist_ok=True, parents=True)

    dataset = data['dataset']
    times = dataset.times
    subjects = dataset.subjects
    reads = dataset.get_reads()

    if sparsity_prior is None:
        sparsity_prior = 0.5/num_assemblages
    if sparsity_power is None:
        num_reads = 0
        for t in times:
            for s in subjects:
                num_reads += reads[t][s].sum()
        sparsity_power = 0.005 * num_reads
    if perturbation_prior is None:
        perturbation_prior = 0.5/num_assemblages

    seeds = np.arange(n_seeds)
    for seed in seeds:
        run_inference_seed(data,
                          runs_outpath,
                          seed,
                          n_epochs,
                          learning_rate,
                          num_assemblages,
                          sparsity_prior,
                          sparsity_power,
                          anneal_prior,
                          process_variance_prior,
                          perturbation_prior,
                          use_contamination,
                          use_sparsity,
                          use_kmeans_init,
                          device)

    # get best model and save posterior summary
    save_best_model_posterior_summary(runs_outpath, seeds, outpath)
