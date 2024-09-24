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


def run_case(basepath, case, fold):
    datapath = basepath / "holdout_data" / case
    outpathbase = basepath / "gmm_basic" / case / f"Fold_F{fold}"
    outpathbase.mkdir(exist_ok=True, parents=True)

    reads = pickle_load(datapath / f"train_F{fold}.pkl")

    klist = np.arange(2,51)
    for ncomm in klist:
        print(f"...fitting k = {ncomm}")
        outpath = outpathbase / f"K_{ncomm}"
        outpath.mkdir(exist_ok=True, parents=True)

        model = BasicGaussianMixture(ncomm)
        model.fit_model(reads)
        results = model.get_params()
        #* save results
        pickle_save(outpath / RESULT_FILE, results)
        pickle_save(outpath / MODEL_FILE, model)
    print(f"done case: {case}, fold: {fold}")


def get_cases(rootpath):
    all_cases = []

    nfolds = 5
    dsets = [get_human_timeseries_dataset, get_mouse_diet_perturbations_dataset]
    names = ['Human', 'Mouse']

    for dset, name in zip(dsets, names):
        reads, num_otus, times, subjects, dataset = dset(rootpath=rootpath)
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

    np.random.seed(0)

    all_cases = get_cases(datapath)
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
