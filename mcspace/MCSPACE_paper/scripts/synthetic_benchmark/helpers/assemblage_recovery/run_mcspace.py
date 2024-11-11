import os
os.environ["OMP_NUM_THREADS"] = "40" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "40" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "40" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "40" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "40" # export NUMEXPR_NUM_THREADS=6

import numpy as np
import torch
from mcspace.model import MCSPACE
from mcspace.trainer import train_model, train_and_trace_model
from mcspace.data_utils import get_data
from mcspace.utils import get_device, pickle_load, pickle_save, MODEL_FILE
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import time
mpl.use('agg')


def run_case(outpathbase, datapathbase, case, seed, base_sample):
    device = get_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    outpath = outpathbase / "mcspace" / base_sample / case / f"seed_{seed}"
    outpath.mkdir(exist_ok=True, parents=True)

    sim_dataset = pickle_load(datapathbase / base_sample / f"data_{case}.pkl")
    reads = sim_dataset['reads']
    num_otus = reads[0]['s1'].shape[1]
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
                    use_contamination=True,
                    contamination_clusters=data['group_garbage_clusters']
                )
    model.kmeans_init(data, seed=seed)
    model.to(device)

    num_epochs = 5000
    ELBOs = train_model(model, data, num_epochs, anneal_prior=True)
    torch.save(model, outpath / MODEL_FILE)


def get_cases(npart_cases, nreads_cases, dsets, base_sample):
    all_cases = []
    for ds in dsets:
        # vary particles
        for npart in npart_cases:
            case = f"D{ds}_P{npart}_Rdefault_B{base_sample}"
            all_cases.append(case)

        # vary reads
        for nreads in nreads_cases:
            case = f"D{ds}_Pdefault_R{nreads}_B{base_sample}"
            all_cases.append(case)
    return all_cases


def main(rootdir, outdir, run_idx, base_sample):
    st = time.time()

    rootpath = Path(rootdir)
    outpathbase = Path(outdir) / "assemblage_recovery"
    datapathbase = Path(outdir) / "semisyn_data"

    npart_cases = [5000, 2500, 1000, 500, 250]
    nreads_cases = [5000, 2500, 1000, 500, 250]
    dsets = np.arange(10)

    all_cases = get_cases(npart_cases, nreads_cases, dsets, base_sample)
    print(len(all_cases), "cases")
    case = all_cases[run_idx]

    print(f"running case: {case}")
    for seed in range(10):
        run_case(outpathbase, datapathbase, case, seed, base_sample) 
    print("***DONE***")
    et = time.time()
    elapsed_time = et - st
    print(elapsed_time)


def run_all(rootdir, outdir):
    base_sample = 'Human'
    ncases = 100
    for ridx in range(ncases):
        main(rootdir, outdir, ridx, base_sample)


if __name__ == "__main__":
    # 100 human cases
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", dest='rootpath', help='root path')
    parser.add_argument("-o", dest='outpath', help='output path')
    parser.add_argument("-run_all", dest='run_all', help='option to run all cases', action='store_true')    
    parser.add_argument("-idx", type=int, dest='idx', help='run number (200 total cases for human; 250 for mouse datasets)')
    parser.add_argument("-dset", dest='dset', help='dataset case (Human or Mouse)')
    args = parser.parse_args()
    if args.run_all is True:
        print("RUNNING ALL CASES")
        run_all(args.rootpath, args.outpath)
    else:
        print(f"Running {args.dset} case {args.idx}")
        main(args.rootpath, args.outpath, args.idx, args.dset)
        