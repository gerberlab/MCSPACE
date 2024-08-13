import os
os.environ["OMP_NUM_THREADS"] = "40" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "40" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "40" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "40" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "40" # export NUMEXPR_NUM_THREADS=6

import numpy as np
from mcspace.utils import pickle_load, pickle_save, RESULT_FILE, MODEL_FILE
from mcspace.comparators.comparator_models import DirectionalGaussianMixture
from pathlib import Path
import time 
from mcspace.data_utils import get_human_timeseries_dataset, get_mouse_diet_perturbations_dataset


def run_case(basepath, case, fold):
    datapath = basepath / "holdout_data" / case
    outpathbase = basepath / "gmm_two_dim" / case / f"Fold_F{fold}"
    outpathbase.mkdir(exist_ok=True, parents=True)

    reads = pickle_load(datapath / f"train_F{fold}.pkl")

    klist = [4, 6, 8, 10] #! model does not converge for K>10
    for ncomm in klist:
        print(f"...fitting k = {ncomm}")
        outpath = outpathbase / f"K_{ncomm}"
        outpath.mkdir(exist_ok=True, parents=True)

        model =DirectionalGaussianMixture(ncomm, dim=2)
        model.fit_model(reads)
        results = model.get_params()
        #* save results
        pickle_save(outpath / RESULT_FILE, results)
        pickle_save(outpath / MODEL_FILE, model)
    print(f"done case: {case}, fold: {fold}")


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


def main(rootdir, run_idx):
    # TODO: add rootpath as command line argument
    rootpath = Path(rootdir)
    basepath = rootpath / "paper_cluster" / "cross_validation"
    np.random.seed(0)

    all_cases = get_cases()
    case, fold = all_cases[run_idx]

    print(f"running case: {case}, fold: {fold}")
    run_case(basepath, case, fold)   


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", dest='directory', help='root path')
    parser.add_argument("--idx", type=int, dest='idx', help='run number')
    args = parser.parse_args()
    main(args.directory, args.idx)
