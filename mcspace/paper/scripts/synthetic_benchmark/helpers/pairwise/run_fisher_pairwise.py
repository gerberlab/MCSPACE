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
from sklearn.metrics import auc
import torch
from pathlib import Path
from mcspace.utils import pickle_load, pickle_save
import time
from pairwise_utils import get_gt_assoc, calc_auc
import warnings
warnings.filterwarnings("ignore")

def get_sig_assoc_fisher(data_df, threshold=0.005):
    sig=0.05
    databin = data_df > threshold # LxO df of T/F values
    data = databin.T 

    npart, notus = data.shape
    pvalue_a=[]
    or_a=[]
    for i in range(1,notus):
        for j in range(i):
            table=pd.crosstab(data.T.iloc[i] > 0, data.T.iloc[j] > 0)
            # print(table)
            if table.shape == (2,2):
                oddsratio,p_value=stats.fisher_exact(table, alternative = 'greater')
                pvalue_a.append(p_value)
                or_a.append(np.log2(oddsratio))
            else:
                pvalue_a.append(np.nan)
                or_a.append(0)

    # multiple testing correction
    # deal with nans
    pvalue_a = np.array(pvalue_a)
    nonnaninds = ~np.isnan(pvalue_a)
    pvalue_a_clean = pvalue_a[nonnaninds]
    _,pvalue_c_clean,_,_=multipletests(pvalue_a_clean,alpha=sig,method='fdr_bh')
    pvalue_c = np.ones(len(pvalue_a))
    pvalue_c[nonnaninds] = pvalue_c_clean

    #reshape into matrix format
    mat_pv=np.zeros(shape=(data.shape[1],data.shape[1]))
    learned_assoc=np.zeros(shape=(data.shape[1],data.shape[1]))
    mat_or=np.zeros(shape=(data.shape[1],data.shape[1]))
    mat_sig= pd.DataFrame(index=data.T.index, columns=data.T.index)
    cnt=0
    for i in np.arange(1,data.shape[1]):
        for i2 in np.arange(i): 
            mat_pv[i,i2]=pvalue_c[cnt]
            mat_or[i,i2]=or_a[cnt]
            if pvalue_c[cnt] < sig:
                mat_sig.iloc[i,i2] = str('x')
                if (mat_or[i,i2] > 0):
                    learned_assoc[i,i2] = 1
            else:
                mat_sig.iloc[i,i2] = str('')
            cnt+=1
    index_p=data.T.index
    to_plot=pd.DataFrame(mat_or+mat_or.T,index=index_p, columns=index_p)
    mat_sig = mat_sig.T.replace(np.nan,'') + mat_sig.replace(np.nan,'')
    learned_assoc = learned_assoc + learned_assoc.T

    mat_pv[np.isnan(mat_pv)] = 1.0
    mat_pv = mat_pv + mat_pv.T
    for i in range(data.shape[1]):
        mat_pv[i,i] = 1.0
    return learned_assoc, to_plot, mat_sig, mat_pv




def get_data_df_for_fisher(reads):
    npart, notus = reads.shape
    index = [f"OTU_{i+1}" for i in range(notus)]
    cols = [f"P{i+1}" for i in range(npart)]
    #* convert to relative abundance
    relabun = reads/(reads.sum(axis=1, keepdims=True))
    data_df = pd.DataFrame(data=relabun.T, index=index, columns=cols)
    return data_df


def eval_fisher_pairwise_auc(reads, theta):
    otu_threshold = 0.005
    nthres = 100
    gt_assoc = get_gt_assoc(theta, otu_threshold=otu_threshold)
    # get fisher data and pvalues
    data_df = get_data_df_for_fisher(reads)
    learned_assoc, to_plot, mat_sig, pvals = get_sig_assoc_fisher(data_df, threshold=otu_threshold)
    # take 1 - pvals for prob of association
    fish_pvals = 1.0 - pvals
    auc_val, true_pos, false_pos, true_neg, false_neg = calc_auc(gt_assoc, fish_pvals, nthres)
    return auc_val


def combine_samples(reads):
    times = list(reads.keys())
    subjs = list(reads[times[0]].keys())
    combreads = []
    for t in times:
        for s in subjs:
            combreads.append(reads[t][s])
    allreads = np.concatenate(combreads, axis=0)    
    return allreads


def evaluate_case(datapath, results, ds, nk, npart, nreads, gweight, nsubj, base_sample):
    case = f"D{ds}_K{nk}_P{npart}_R{nreads}_G{gweight}_B{base_sample}_S{nsubj}"

    # load ground truth data 
    gtdata = pickle_load(datapath / f"data_{case}.pkl")

    reads = combine_samples(gtdata['reads']) #[0]['s1']
    gt_theta = gtdata['theta']

    # eval fisher
    aucval = eval_fisher_pairwise_auc(reads, gt_theta)

    results['model'].append('fisher')
    results['base_sample'].append(base_sample)
    results['number particles'].append(npart)
    results['read depth'].append(nreads)
    results['number clusters'].append(nk)
    results['garbage weight'].append(gweight)
    results['number subjects'].append(nsubj)
    results['dataset'].append(ds)
    results['auc'].append(aucval)
    return results


def main(rootdir, outdir):
    rootpath = Path(rootdir)
    # basepath = rootpath / "paper_cluster" / "pairwise"

    outpath =  rootpath / "results" / "pairwise" / "fisher_results"
    outpath.mkdir(exist_ok=True, parents=True)

    st = time.time()
    
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

    for base_sample in ['Mouse', 'Human']:
        #* cases
        if base_sample == 'Mouse':
            nsubj_cases = [1,3,5,7,10]
        npart_cases = [10000, 5000, 1000, 500, 100]
        nreads_cases = [10000, 5000, 1000, 500, 100]
        nclust_cases = [5, 10, 15, 20, 25]
        pgarb_cases = [0.0, 0.025, 0.05, 0.075, 0.1]
        dsets = np.arange(10)

        datapath = Path(outdir) / "semisyn_data" / base_sample

        for ds in dsets:
            print(f"analyzing dataset {ds}...")

            for nk in nclust_cases:
                results = evaluate_case(datapath, results, ds, nk, "default", "default", "default", "default", base_sample)

            for npart in npart_cases:
                results = evaluate_case(datapath, results, ds, "default", npart, "default", "default", "default", base_sample)

            for nreads in nreads_cases:
                results = evaluate_case(datapath, results, ds, "default", "default", nreads, "default", "default", base_sample)

            for gweight in pgarb_cases:
                results = evaluate_case(datapath, results, ds, "default", "default", "default", gweight, "default", base_sample)
            
            if base_sample == "Mouse":
                for nsubj in nsubj_cases:
                    results = evaluate_case(datapath, results, ds, "default", "default", "default", "default", nsubj, base_sample)

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
