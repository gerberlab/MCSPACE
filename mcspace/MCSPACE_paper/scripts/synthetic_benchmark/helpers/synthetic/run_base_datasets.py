import numpy as np
import torch
from mcspace.model import MCSPACE
from mcspace.trainer import train_model
from mcspace.data_utils import get_data, get_human_timeseries_dataset, get_mouse_diet_perturbations_dataset
from mcspace.utils import get_device, pickle_load, pickle_save, get_summary_results, MODEL_FILE
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
mpl.use('agg')


def run_case(basepath, case, reads, num_otus, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = get_device()
    outpath = basepath / "base_run" / case
    outpath.mkdir(exist_ok=True, parents=True)

    data = get_data(reads, device)

    num_assemblages = 100
    times = list(reads.keys())
    subjects = list(reads[times[0]].keys())
    perturbed_times = []
    perturbation_prior = None
    sparsity_prior = 0.5/num_assemblages

    num_reads = 0
    for t in times:
        for s in subjects:
            num_reads += reads[t][s].sum()
    sparsity_prior_power = 0.005*num_reads 

    process_var_prior = None
    add_process_var=False

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
                    use_sparse_weights=True,
                    use_contamination=True,
                    contamination_clusters=data['group_garbage_clusters']
                    )
    model.kmeans_init(data)
    model.to(device)

    num_epochs = 5000
    ELBOs = train_model(model, data, num_epochs, anneal_prior=True)
    torch.save(model, outpath / MODEL_FILE)

    pert_bf, beta, theta, pi_garb, mean_loss = get_summary_results(model, data, gamma_percentile=0.95)
    print("THETA 95 size = ", theta.shape)
    gclust = data['group_garbage_clusters'][0].cpu().detach().clone().numpy()
    res = {'pert_bf': pert_bf, 
        'beta': beta, 
        'theta': theta,
        'pi_garb': pi_garb,
        'loss': mean_loss,
        'garbage_cluster': gclust}
    pickle_save(outpath / "results_95.pkl", res)

    pert_bf, beta, theta, pi_garb, mean_loss = get_summary_results(model, data, gamma_percentile=0.99)
    print("THETA 99 size = ", theta.shape)
    gclust = data['group_garbage_clusters'][0].cpu().detach().clone().numpy()
    res = {'pert_bf': pert_bf, 
        'beta': beta, 
        'theta': theta,
        'pi_garb': pi_garb,
        'loss': mean_loss,
        'garbage_cluster': gclust}
    pickle_save(outpath / "results_99.pkl", res)


    # plot losses
    fig, ax = plt.subplots()
    ax.plot(ELBOs)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ELBO loss")
    plt.savefig(outpath / "ELBO_loss.png")
    plt.close()
    print(f"DONE: case={case}")


def main(rootdir):
    seed = 42

    rootpath = Path(rootdir)
    basepath = rootpath / "scripts" / "synthetic_benchmark" / "helpers" / "synthetic"

    name = 'Human'
    reads, num_otus, times, subjects, dataset = get_human_timeseries_dataset(rootpath=rootpath / "datasets")
    t = times[0]
    counts = {0: reads[t]} # take first timepoint
    run_case(basepath, name, counts, num_otus, seed=seed)
    print("***ALL DONE***")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", dest='rootdir', help='project directory path')
    args = parser.parse_args()
    main(args.rootdir)
