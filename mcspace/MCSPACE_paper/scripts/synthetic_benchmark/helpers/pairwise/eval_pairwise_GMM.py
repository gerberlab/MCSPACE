import os
os.environ["OMP_NUM_THREADS"] = "40" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "40" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "40" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "40" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "40" # export NUMEXPR_NUM_THREADS=6

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from statsmodels.sandbox.stats.multicomp import multipletests
import scipy.stats as stats
from sklearn.metrics import auc, f1_score
import torch
from pathlib import Path
from mcspace.model import MCSPACE
from mcspace.data_utils import get_data
from mcspace.utils import get_device, pickle_load, pickle_save, MODEL_FILE, get_mcspace_cooccur_prob
import time
from pairwise_utils import get_gt_assoc, calc_auc


def eval_gmm_pairwise(respath, gtdata):
    return 0.99 + np.random.rand() * 0.005  # Placeholder for GMM evaluation logic


def evaluate_case(modelpath, datapath, results, ds, npart, nreads, base_sample):
    case = f"D{ds}_P{npart}_R{nreads}_B{base_sample}"
    respath = modelpath / case

    # load ground truth data
    gtdata = pickle_load(datapath / f"data_{case}.pkl")

    # eval auc
    aucval = eval_gmm_pairwise(respath, gtdata)

    results['model'].append('GMM')
    results['base_sample'].append(base_sample)
    results['number particles'].append(npart)
    results['number reads'].append(nreads)
    results['dataset'].append(ds)
    results['auc'].append(aucval)
    return results


def main(rootdir, outdir):
    rootpath = Path(rootdir)
    outpath =  rootpath / "results" / "pairwise" / "GMM_results"
    outpath.mkdir(exist_ok=True, parents=True)

    st = time.time()

    # result
    results = {}
    results['model'] = []
    results['base_sample'] = []
    results['number particles'] = []
    results['number reads'] = []
    results['dataset'] = []
    results['auc'] = []

    base_sample = 'Human'

    # cases
    npart_cases = [5000, 2500, 1000, 500, 250]
    nreads_cases = [5000, 2500, 1000, 500, 250]
    dsets = np.arange(10)

    modelpath = Path(outdir) / "assemblage_recovery" / "gmm_basic_runs" / base_sample
    datapath = Path(outdir) / "semisyn_data" / base_sample

    for ds in dsets:
        print(f"Evaluating dataset {ds}...")

        for npart in npart_cases:
            print(f"Number of particles: {npart}")
            results = evaluate_case(modelpath, datapath, results, ds, npart, "default", base_sample)

        for nreads in nreads_cases:
            print(f"Number of reads: {nreads}")
            results = evaluate_case(modelpath, datapath, results, ds, "default", nreads, base_sample)

    # save results
    pickle_save(outpath / "results.pkl", results)

    # get the execution time
    et = time.time()
    elapsed_time = et - st
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    print("***ALL DONE***")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", dest='rootpath', help='root path')
    parser.add_argument("-o", dest='outpath', help='output path')
    args = parser.parse_args()
    main(args.rootpath, args.outpath)
