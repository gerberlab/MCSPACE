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
    

def evaluate_case(modelpath, datapath, results, ds, nk, npart, nreads, gweight, nsubj, base_sample):
    case = f"D{ds}_K{nk}_P{npart}_R{nreads}_G{gweight}_B{base_sample}_S{nsubj}"

    # load ground truth data 
    gtdata = pickle_load(datapath / f"data_{case}.pkl")
    reads = gtdata['reads'][0]['s1'] # only to get num_otus
    gt_theta = gtdata['theta']

    # eval ecosim
    ecosim_results = modelpath / f"results_bindata_{case}.csv"
    aucval = eval_ecosim_pairwise_auc(ecosim_results, reads, gt_theta)

    results['model'].append('SIM9')
    results['base_sample'].append(base_sample)
    results['number particles'].append(npart)
    results['read depth'].append(nreads)
    results['number clusters'].append(nk)
    results['garbage weight'].append(gweight)
    results['number subjects'].append(nsubj)
    results['dataset'].append(ds)
    results['auc'].append(aucval)
    return results


def main(rootdir):
    rootpath = Path(rootdir) #! change

    basepath = rootpath / "paper_cluster" / "pairwise" 

    outpath = basepath / "ecosim_eval_results"
    outpath.mkdir(exist_ok=True, parents=True)

    st = time.time()

    #* cases
    npart_cases = [10000, 5000, 1000, 500, 100]
    nreads_cases = [10000, 5000, 1000, 500, 100]
    nclust_cases = [5, 10, 15, 20, 25]
    pgarb_cases = [0.0, 0.025, 0.05, 0.075, 0.1]
    dsets = np.arange(10)


    # result
    results = {}
    results['model'] = []
    results['base_sample'] = []
    results['number particles'] = []
    results['read depth'] = []
    results['number clusters'] = []
    results['garbage weight'] = []
    results['number subjects'] = []
    results['dataset'] = []
    results['auc'] = []

    for base_sample in ['Mouse']: #, 'Human']:
        #* cases
        if base_sample == 'Mouse':
            nsubj_cases = [1,3,5,7,10]
        npart_cases = [10000, 5000, 1000, 500, 100]
        nreads_cases = [10000, 5000, 1000, 500, 100]
        nclust_cases = [5, 10, 15, 20, 25]
        pgarb_cases = [0.0, 0.025, 0.05, 0.075, 0.1]
        dsets = np.arange(10)
        
        modelpath = basepath / "ecosimR_results" / base_sample
        datapath = rootpath / "paper_cluster" / "semi_synthetic_data" / "semisyn_data" / base_sample


        for ds in dsets:
            print(f"analyzing dataset {ds}...")
            # evaluate_case(modelpath, datapath, results, ds, nk, npart, nreads, gweight, nsubj, base_sample)

            for nk in nclust_cases:
                results = evaluate_case(modelpath, datapath, results, ds, nk, "default", "default", "default", "default", base_sample)

            for npart in npart_cases:
                results = evaluate_case(modelpath, datapath, results, ds, "default", npart, "default", "default", "default", base_sample)

            for nreads in nreads_cases:
                results = evaluate_case(modelpath, datapath, results, ds, "default", "default", nreads, "default", "default", base_sample)

            for gweight in pgarb_cases:
                results = evaluate_case(modelpath, datapath, results, ds, "default", "default", "default", gweight, "default", base_sample)
            
            if base_sample == "Mouse":
                for nsubj in nsubj_cases:
                    results = evaluate_case(modelpath, datapath, results, ds, "default", "default", "default", "default", nsubj, base_sample)

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
    parser.add_argument("--directory", dest='directory', help='root path')
    args = parser.parse_args()
    main(args.directory)
