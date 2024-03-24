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
from mcspace.data_utils import get_data
from mcspace.utils import get_device, pickle_load, pickle_save, get_summary_stats, MODEL_FILE
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import mcspace.visualization as vis
import time
mpl.use('agg')


def run_case(basepath, datapath, case, seed, base_sample):
    device = get_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    outpath = basepath / "mcspace_runs" / base_sample / case / f"seed_{seed}"
    outpath.mkdir(exist_ok=True, parents=True)

    sim_dataset = pickle_load(datapath / f"data_{case}.pkl")
    reads = sim_dataset['reads']
    num_otus = reads[0]['s1'].shape[1]
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
    print(f"DONE: case={case}")


def get_cases(npart_cases, nreads_cases, nclust_cases, dsets, base_sample):
    all_cases = []

    for ds in dsets:
        # vary number of clusters
        for nk in nclust_cases:
            case = f"D{ds}_K{nk}_Pdefault_Rdefault_B{base_sample}"
            all_cases.append(case)

        # vary particles
        for npart in npart_cases:
            case = f"D{ds}_Kdefault_P{npart}_Rdefault_B{base_sample}"
            all_cases.append(case)

        # vary reads
        for nreads in nreads_cases:
            case = f"D{ds}_Kdefault_Pdefault_R{nreads}_B{base_sample}"
            all_cases.append(case)
    return all_cases


def main(run_idx):
    st = time.time()

    # seed = 0
    base_sample = 'Mouse' #! to do, loop over...
    rootpath = Path("./") #! path to mcspace

    basepath = rootpath / "paper" / "assemblage_recovery"
    datapath = rootpath / "paper" / "semi_synthetic" / "semisyn_data" / base_sample

    npart_cases = [10000, 5000, 1000, 500, 100]
    nreads_cases = [10000, 5000, 1000, 500, 100]
    nclust_cases = [5, 10, 15, 20, 25]
    dsets = np.arange(10)

    all_cases = get_cases(npart_cases, nreads_cases, nclust_cases, dsets, base_sample)
    print(len(all_cases), "cases")
    case = all_cases[run_idx]

    print(f"running case: {case}")
    for seed in range(5):
        run_case(basepath, datapath, case, seed, base_sample) 
    print("***DONE***")
    et = time.time()
    elapsed_time = et - st
    print(elapsed_time)


if __name__ == "__main__":
    import sys
    run_idx = int(sys.argv[1])
    print(f"running case ID: {run_idx}")
    main(run_idx=0)

    # for i in range(150):
    #     main(i)
    # print("***ALL DONE***")
