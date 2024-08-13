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


def evaluate_case(modelpath, datapath, results, ds, nk, npart, nreads, base_sample):
    case = f"D{ds}_K{nk}_P{npart}_R{nreads}_B{base_sample}"

    # load ground truth data 
    gtdata = pickle_load(datapath / f"data_{case}.pkl")
    reads = gtdata['reads'][0]['s1']
    gt_theta = gtdata['theta']

    # eval ecosim
    ecosim_results = modelpath / f"results_{case}.csv"
    aucval = eval_ecosim_pairwise_auc(ecosim_results, reads, gt_theta)

    results['model'].append('SIM9')
    results['base_sample'].append(base_sample)
    results['number particles'].append(npart)
    results['read depth'].append(nreads)
    results['number clusters'].append(nk)
    results['dataset'].append(ds)
    results['auc'].append(aucval)
    return results


def main():
    rootpath = Path("./")

    basepath = rootpath / "paper" / "pairwise" 

    outpath = basepath / "ecosim_eval_results"
    outpath.mkdir(exist_ok=True, parents=True)

    st = time.time()

    #* cases
    npart_cases = [10000, 5000, 1000, 500, 100]
    nreads_cases = [10000, 5000, 1000, 500, 100]
    nclust_cases = [5, 10, 15, 20, 25]
    dsets = np.arange(10)

    results = {}
    results['model'] = []
    results['base_sample'] = []
    results['number particles'] = []
    results['read depth'] = []
    results['number clusters'] = []
    results['dataset'] = []
    results['auc'] = []

    for base_sample in ['Mouse', 'Human']:
        modelpath = basepath / "ecosimR_results" / base_sample
        datapath = rootpath / "paper" / "semi_synthetic" / "semisyn_data" / base_sample

        for dset in dsets:
            print(f"analyzing dataset {dset}...")

            # vs number clusters
            for nk in nclust_cases:
                results = evaluate_case(modelpath, datapath, results, dset, nk, "default", "default", base_sample)

            # vs number particles
            for npart in npart_cases:
                results = evaluate_case(modelpath, datapath, results, dset, "default", npart, "default", base_sample)

            # vs number reads
            for nreads in nreads_cases:
                results = evaluate_case(modelpath, datapath, results, dset, "default", "default", nreads, base_sample)
        print(f"...done {base_sample} samples")

    # save results
    pickle_save(outpath / "results.pkl", results)

    # get the execution time
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
    print("***ALL DONE***")


if __name__ == "__main__":
    main()
