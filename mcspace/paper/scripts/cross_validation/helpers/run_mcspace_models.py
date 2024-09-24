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
from mcspace.data_utils import get_data, get_human_timeseries_dataset, get_mouse_diet_perturbations_dataset
from mcspace.utils import get_device, pickle_load, pickle_save, MODEL_FILE
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import mcspace.visualization as vis
mpl.use('agg')


def run_case(basepath, case, fold):
    device = get_device()
    datapath = basepath / "holdout_data" / case
    outpath = basepath / f"mcspace_runs" / case / f"Fold_F{fold}"
    outpath.mkdir(exist_ok=True, parents=True)

    reads_single = pickle_load(datapath / f"train_F{fold}.pkl")
    num_otus = reads_single.shape[1]
    reads = {}
    reads[0] = {}
    reads[0]['s1'] = reads_single
    data = get_data(reads, device)

    num_assemblages = 50
    times = list(reads.keys())
    subjects = list(reads[times[0]].keys())
    perturbed_times = []
    perturbation_prior = None
    sparsity_prior = 0.5/num_assemblages

    num_reads = 0
    for t in times:
        for s in subjects:
            num_reads += reads[t][s].sum()
    sparsity_prior_power = 0.001*num_reads 

    process_var_prior = None
    add_process_var=False

    # get model instance
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
                    contamination_clusters=data['group_garbage_clusters'])
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


def get_cases(datapath):
    all_cases = []

    nfolds = 5
    dsets = [get_human_timeseries_dataset, get_mouse_diet_perturbations_dataset]
    names = ['Human', 'Mouse']

    for dset, name in zip(dsets, names):
        reads, num_otus, times, subjects, dataset = dset(rootpath=datapath)
        for t in times:
            for s in subjects:
                for fold in range(nfolds):
                    case = f"{name}_{t}_{s}"
                    x = (case, fold)
                    all_cases.append(x)
    return all_cases


def main(rootdir, outdir, run_idx):
    # TODO: add rootpath as command line argument
    rootpath = Path(rootdir)
    datapath = rootpath / "datasets"
    basepath = Path(outdir) / "cross_validation"

    torch.manual_seed(0)
    np.random.seed(0)

    all_cases = get_cases(datapath)
    print(len(all_cases), "cases")
    case, fold = all_cases[run_idx]

    print(f"running case: {case}, fold: {fold}")

    run_case(basepath, case, fold)


def run_all(rootdir, outdir):
    ncases = 130
    for ridx in range(ncases):
        main(rootdir, outdir, ridx)


if __name__ == "__main__":
    # 130 cases
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", dest='rootpath', help='root path')
    parser.add_argument("-o", dest='outpath', help='output path')
    parser.add_argument("-run_all", dest='run_all', help='option to run all cases', action='store_true')
    parser.add_argument("-idx", type=int, dest='idx', help='run number (130 total cases)')
    args = parser.parse_args()
    if args.run_all is True:
        print("RUNNING ALL 130 CASES")
        run_all(args.rootpath, args.outpath)
    else:
        print(f"Running case {args.idx} of 130")
        main(args.rootpath, args.outpath, args.idx)
