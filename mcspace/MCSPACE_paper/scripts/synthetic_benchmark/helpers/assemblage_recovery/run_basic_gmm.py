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

    klist = np.arange(2,101)
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
    ncases = 100
    base_sample = 'Human'
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
