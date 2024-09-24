import os
os.environ["OMP_NUM_THREADS"] = "40" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "40" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "40" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "40" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "40" # export NUMEXPR_NUM_THREADS=6

import numpy as np
from mcspace.utils import pickle_load, pickle_save, RESULT_FILE, MODEL_FILE
from mcspace.comparators.comparator_models import BasicGaussianMixture
from pathlib import Path
import time 
from mcspace.data_utils import get_human_timeseries_dataset, get_mouse_diet_perturbations_dataset


def combine_samples(reads):
    times = list(reads.keys())
    subjs = list(reads[times[0]].keys())
    combreads = []
    for t in times:
        for s in subjs:
            combreads.append(reads[t][s])
    allreads = np.concatenate(combreads, axis=0)    
    return allreads


def run_case(basepath, datapathbase, case, seed, base_sample):
    np.random.seed(seed)

    outpathbase = basepath / "gmm_basic_runs" / base_sample / case
    outpathbase.mkdir(exist_ok=True, parents=True)

    sim_dataset = pickle_load(datapathbase / base_sample / f"data_{case}.pkl")
    counts = sim_dataset['reads']
    reads = combine_samples(counts) #[0]['s1']

    klist = np.arange(2,51)
    for ncomm in klist:
        print(f"...fitting k = {ncomm}")
        outpath = outpathbase / f"K_{ncomm}_seed_{seed}"
        outpath.mkdir(exist_ok=True, parents=True)

        model = BasicGaussianMixture(ncomm)
        model.fit_model(reads)
        results = model.get_params()
        #* save results
        pickle_save(outpath / RESULT_FILE, results)
        pickle_save(outpath / MODEL_FILE, model)
    print(f"done case: {case}")


def get_cases(npart_cases, nreads_cases, nclust_cases, pgarb_cases, nsubj_cases, dsets, base_sample):
    all_cases = []

    for ds in dsets:
        # vary number of subjects for mouse
        if base_sample == 'Mouse':
            for nsubj in nsubj_cases:
                case = f"D{ds}_Kdefault_Pdefault_Rdefault_Gdefault_B{base_sample}_S{nsubj}"
                all_cases.append(case)

        # vary number of clusters
        for nk in nclust_cases:
            case = f"D{ds}_K{nk}_Pdefault_Rdefault_Gdefault_B{base_sample}_Sdefault"
            all_cases.append(case)

        # vary particles
        for npart in npart_cases:
            case = f"D{ds}_Kdefault_P{npart}_Rdefault_Gdefault_B{base_sample}_Sdefault"
            all_cases.append(case)

        # vary reads
        for nreads in nreads_cases:
            case = f"D{ds}_Kdefault_Pdefault_R{nreads}_Gdefault_B{base_sample}_Sdefault"
            all_cases.append(case)

        for gpi in pgarb_cases:
            case = f"D{ds}_Kdefault_Pdefault_Rdefault_G{gpi}_B{base_sample}_Sdefault"
            all_cases.append(case)

    return all_cases



def main(rootdir, outdir, run_idx, base_sample):
    st = time.time()

    # rootpath = Path(rootdir)
    # basepath = rootpath / "paper_cluster" / "assemblage_recovery"
    # datapath = rootpath / "paper_cluster" / "semi_synthetic_data" / "semisyn_data" / base_sample

    rootpath = Path(rootdir)
    outpathbase = Path(outdir) / "assemblage_recovery"
    datapathbase = Path(outdir) / "semisyn_data"


    if base_sample == 'Human':
        nsubj_cases = [1]
    else:
        nsubj_cases = [1,3,5,7,10]
    npart_cases = [10000, 5000, 1000, 500, 100]
    nreads_cases = [10000, 5000, 1000, 500, 100]
    nclust_cases = [5, 10, 15, 20, 25]
    pgarb_cases = [0.0, 0.025, 0.05, 0.075, 0.1]
    dsets = np.arange(10)

    all_cases = get_cases(npart_cases, nreads_cases, nclust_cases, pgarb_cases, nsubj_cases, dsets, base_sample)
    print(len(all_cases), "cases")
    case = all_cases[run_idx]

    print(f"running case: {case}")
    for seed in range(5):
        run_case(outpathbase, datapathbase, case, seed, base_sample) 
    print("***DONE***")
    et = time.time()
    elapsed_time = et - st
    print(elapsed_time)


def run_all(rootdir, outdir):
    for ncases, base_sample in zip([200, 250], ['Human', 'Mouse']):
        for ridx in range(ncases):
            main(rootdir, outdir, ridx, base_sample)

# if __name__ == "__main__":
#     # 200 human cases
#     # 250 mouse cases
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--directory", dest='directory', help='root path')
#     parser.add_argument("--idx", type=int, dest='idx', help='run number')
#     parser.add_argument("--dset", dest='dset', help='dataset case (Human or Mouse)')
#     args = parser.parse_args()
#     main(args.directory, args.idx, args.dset)


if __name__ == "__main__":
    # 200 human cases
    # 250 mouse cases
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
