import os
os.environ["OMP_NUM_THREADS"] = "40" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "40" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "40" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "40" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "40" # export NUMEXPR_NUM_THREADS=6

import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.sandbox.stats.multicomp import multipletests
import scipy.stats as stats
from sklearn.metrics import auc
from pathlib import Path
from mcspace.utils import pickle_load, pickle_save
from pairwise_utils import get_gt_assoc, calc_auc
import time


def extract_ecosim_pvals(res, num_otus):
    restab = res.pivot(index="Var1", columns="Var2", values="zpadjusted")
    idx = [f"ASV{i+1}" for i in range(num_otus)]
    restab = restab.reindex(idx)
    restab = restab.T.reindex(idx).T
    # matrix is triangular, can symetrize to make sure no nans are wrongly on one side
    data = restab.values
    for i in range(num_otus):
        for j in range(num_otus):
            if (np.isnan(data[i,j])) and not (np.isnan(data[j,i])):
                data[i,j] = data[j,i]
    resfinal = pd.DataFrame(data=data, index=idx, columns=idx)
    resfinal = resfinal.fillna(1.0)
    eco_pvals = 1.0 - resfinal.values
    return eco_pvals


def eval_ecosim_pairwise_auc(ecosim_results, reads, theta):
    _, num_otus = reads.shape
    otu_threshold = 0.005
    nthres = 100
    gt_assoc = get_gt_assoc(theta, otu_threshold=otu_threshold)
    # load ecosim results and extract pvalues
    res = pd.read_csv(ecosim_results)
    eco_pvals = extract_ecosim_pvals(res, num_otus)
    auc_val, true_pos, false_pos, true_neg, false_neg = calc_auc(gt_assoc, eco_pvals, nthres)
    return auc_val
    

def evaluate_case(ecosim_path, datapath, results, ds, npart, nreads, base_sample):
    case = f"D{ds}_P{npart}_R{nreads}_B{base_sample}"

    # load ground truth data 
    gtdata = pickle_load(datapath / f"data_{case}.pkl")
    reads = gtdata['reads'][0]['s1'] # only to get num_otus
    gt_theta = gtdata['theta']

    # eval ecosim
    ecosim_results = ecosim_path / f"results_bindata_{case}.csv"
    aucval = eval_ecosim_pairwise_auc(ecosim_results, reads, gt_theta)

    results['model'].append('SIM9')
    results['base_sample'].append(base_sample)
    results['number particles'].append(npart)
    results['number reads'].append(nreads)
    results['dataset'].append(ds)
    results['auc'].append(aucval)
    return results


def main(rootdir, outdir):
    rootpath = Path(rootdir)
    outpath =  rootpath / "results" / "pairwise" / "ecosim_results"
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
    #* cases
    npart_cases = [5000, 2500, 1000, 500, 250]
    nreads_cases = [5000, 2500, 1000, 500, 250]
    dsets = np.arange(10)
    
    ecosim_path = Path(outdir) / "assemblage_recovery" / "ecosimR_results" / base_sample

    datapath = Path(outdir) / "semisyn_data" / base_sample
    # datapath = rootpath / "paper_cluster" / "semi_synthetic_data" / "semisyn_data" / base_sample


    for ds in dsets:
        print(f"analyzing dataset {ds}...")

        for npart in npart_cases:
            print(f"\tanalyzing number particles={npart}")
            results = evaluate_case(ecosim_path, datapath, results, ds, npart, "default", base_sample)

        for nreads in nreads_cases:
            print(f"\tanalyzing number reads={nreads}")
            results = evaluate_case(ecosim_path, datapath, results, ds, "default", nreads, base_sample)

    # save results
    pickle_save(outpath / f"results.pkl", results)

    # get the execution time
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
    print("***ALL DONE***")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", dest='rootpath', help='root path')
    parser.add_argument("-o", dest='outpath', help='output path')
    args = parser.parse_args()
    main(args.rootpath, args.outpath)
