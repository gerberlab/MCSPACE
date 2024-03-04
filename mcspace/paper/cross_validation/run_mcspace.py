import numpy as np
import torch
from mcspace.model import MCSPACE
from mcspace.trainer import train_model
from mcspace.data_utils import get_data, get_human_timeseries_dataset, get_mouse_diet_perturbations_dataset
from mcspace.utils import get_device, pickle_load, pickle_save, get_summary_stats, MODEL_FILE
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import mcspace.visualization as vis
mpl.use('agg')


def run_case(basepath, case, fold):
    device = get_device()
    datapath = basepath / "holdout_data" / case
    outpath = basepath / "mcspace" / case / f"Fold_F{fold}"
    outpath.mkdir(exist_ok=True, parents=True)

    reads_single = pickle_load(datapath / f"train_F{fold}.pkl")
    num_otus = reads_single.shape[1]
    reads = {}
    reads[0] = {}
    reads[0]['s1'] = reads_single
    data = get_data(reads, device)

    num_assemblages = 30
    times = list(reads.keys())
    subjects = list(reads[times[0]].keys())
    perturbed_times = []
    perturbation_prior = None
    sparsity_prior = 0.5/num_assemblages

    num_reads = 0
    for t in times:
        for s in subjects:
            num_reads += reads[t][s].sum()
    sparsity_prior_power = 0.1*num_reads

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
                    add_process_var)
    model.to(device)

    num_epochs = 5000
    ELBOs = train_model(model, data, num_epochs)
    torch.save(model, outpath / MODEL_FILE)

    # plot losses
    fig, ax = plt.subplots()
    ax.plot(ELBOs)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ELBO loss")
    plt.savefig(outpath / "ELBO_loss.png")
    plt.close()
    print(f"DONE: case={case} fold={fold}")


def get_cases():
    all_cases = []

    nfolds = 5
    dsets = [get_human_timeseries_dataset, get_mouse_diet_perturbations_dataset]
    names = ['Human', 'Mouse']

    for dset, name in zip(dsets, names):
        reads, num_otus, times, subjects, dataset = dset()
        for t in times:
            for s in subjects:
                for fold in range(nfolds):
                    case = f"{name}_{t}_{s}"
                    x = (case, fold)
                    all_cases.append(x)
    return all_cases


def main(run_idx):
    # TODO: add rootpath as command line argument
    rootpath = Path("./")
    basepath = rootpath / "paper" / "cross_validation"

    torch.manual_seed(0)
    np.random.seed(0)

    all_cases = get_cases()
    case, fold = all_cases[run_idx]

    print(f"running case: {case}, fold: {fold}")
    run_case(basepath, case, fold)   


if __name__ == "__main__":
    import sys
    run_idx = int(sys.argv[1])
    print(f"running case ID: {run_idx}")
    main(run_idx)
