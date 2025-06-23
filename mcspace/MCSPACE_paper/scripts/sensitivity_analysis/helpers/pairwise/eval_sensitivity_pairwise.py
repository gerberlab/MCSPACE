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


def get_mcspace_cooccur_binary_prob(model, data, otu_threshold, nsamples=100):
    # time_idx, subj_idx,
    cooccur_prob = 0
    for i in range(nsamples):
        loss, theta, beta, gamma, _ = model(data)
        theta = theta.cpu().clone().detach().numpy()
        beta = beta.cpu().clone().detach().numpy()
        gamma = gamma.cpu().clone().detach().numpy()

        summand = gamma[:,None,None,None,None]*(theta[:,None,None,None,:] > otu_threshold)*(theta[:,None,None,:,None] > otu_threshold)
        prob_sample = (summand.sum(axis=0) > 0.5).astype('float')
        cooccur_prob += prob_sample
    cooccur_prob /= nsamples
    return cooccur_prob


def get_mcspace_cooccur_binary_prob_vs_otu_threshold(model, data, thresholds, nsamples=100):
    #! vectorize over thresholds...
    # time_idx, subj_idx,
    cooccur_prob = 0
    for i in range(nsamples):
        print(f"sample {i} of {nsamples}")
        loss, theta, beta, gamma, _ = model(data)
        theta = theta.cpu().clone().detach().numpy()
        beta = beta.cpu().clone().detach().numpy()
        gamma = gamma.cpu().clone().detach().numpy()

        summand = gamma[:,None,None,None]*(theta[:,None,:,None] > thresholds)*(theta[:,:,None,None] > thresholds)
        prob_sample = (summand.sum(axis=0) > 0.5).astype('float')
        cooccur_prob += prob_sample
    cooccur_prob /= nsamples
    return cooccur_prob


def calc_auc_vs_otu_thresholds(gt_assoc, mcprobs, thresholds):
    notus = gt_assoc.shape[0]
    gta = (gt_assoc[np.triu_indices(notus, k=1)] > 0.5)
    nthres = len(thresholds)

    true_pos = np.zeros((nthres,))
    false_pos = np.zeros((nthres,))
    true_neg = np.zeros((nthres,))
    false_neg = np.zeros((nthres,))

    for i,thres in enumerate(thresholds):
        post_probs = mcprobs[:,:,i]
        pp = post_probs[np.triu_indices(notus, k=1)]
        lta = (pp > thres)
        tp = (gta & lta).sum()
        fp = ((~gta) & lta).sum()
        tn = ((~gta) & (~lta)).sum()
        fn = ((gta) & (~lta)).sum()

        true_pos[i] = tp
        false_pos[i] = fp
        true_neg[i] = tn
        false_neg[i] = fn

    tpr = true_pos/(true_pos + false_neg)
    fpr = false_pos/(false_pos + true_neg)

    auc_val = auc(fpr, tpr)
    return auc_val


def get_mcspace_min_loss_seed(respath, gt_data, seeds):
    device = get_device()
    reads = gt_data['reads']
    data = get_data(reads, device)

    n_samples = 100
    losses = {}
    for seed in seeds:
        modelpath = respath / f"seed_{seed}"
        model = torch.load(modelpath / MODEL_FILE, weights_only=False)
    
        loss_samples = np.zeros(n_samples) 
        for i in range(n_samples):
            loss, _, _, _, _ = model(data)
            loss_samples[i] = loss.item()
        losses[seed] = np.mean(loss_samples)
    
    best_seed = min(losses, key=losses.get)
    return best_seed


def eval_mcspace_pairwise_auc(respath, gt_data, seeds):
    min_seed = get_mcspace_min_loss_seed(respath, gt_data, seeds)
    reads = gt_data['reads'] #![0]['s1']
    gt_theta = gt_data['theta']

    device = get_device()
    otu_threshold = 0.005
    nthres = 100
    data = get_data(reads, device)
    gt_assoc = get_gt_assoc(gt_theta, otu_threshold=otu_threshold)
    # load model
    modelfile = respath / f"seed_{min_seed}" / MODEL_FILE
    model = torch.load(modelfile, weights_only=False)
    # eval mcspace posterior probs
    pvals_all = get_mcspace_cooccur_prob(model, data, otu_threshold, nsamples=1000)
    pvals = pvals_all #[0,0,:,:] #! first time and subject index
    # eval auc
    auc_val, true_pos, false_pos, true_neg, false_neg = calc_auc(gt_assoc, pvals, nthres)
    return auc_val


# def eval_mcspace_pairwise_f1(respath, reads, gt_theta):
#     device = get_device()
#     otu_threshold = 0.005
#     nthres = 100
#     data = get_data(reads, device)
#     gt_assoc = get_gt_assoc(gt_theta, otu_threshold=otu_threshold)
#     # load model
#     modelfile = respath / MODEL_FILE
#     model = torch.load(modelfile)
#     # eval mcspace posterior probs
#     pvals_all = get_mcspace_cooccur_binary_prob(model, data, otu_threshold, nsamples=1000)
#     pvals = pvals_all[0,0,:,:] #! first time and subject index
#     # eval f1
#     notus = gt_assoc.shape[0]
#     gt_pv_upper_tri = gt_assoc[np.triu_indices(notus, k=1)]
#     mcspace_pv_upper_tri = pvals[np.triu_indices(notus, k=1)]

#     true_labels = (gt_pv_upper_tri >= 0.5).astype(int)
#     predicted_labels = (mcspace_pv_upper_tri >= 0.5).astype(int)
#     f1 = f1_score(true_labels, predicted_labels)
#     print('here')
#     return f1


# def eval_mcspace_pairwise_auc_vary_otu_threshold(respath, reads, gt_theta):
#     device = get_device()
#     thresholds = np.linspace(0,1,1000)

#     data = get_data(reads, device)
#     gt_assoc = get_gt_assoc(gt_theta, otu_threshold=0.005)
#     # load model
#     modelfile = respath / MODEL_FILE
#     model = torch.load(modelfile)
#     # eval mcspace posterior probs
#     mcprobs = get_mcspace_cooccur_binary_prob_vs_otu_threshold(model, data, thresholds)
#     # eval auc
#     auc_val = calc_auc_vs_otu_thresholds(gt_assoc, mcprobs, thresholds)
#     return auc_val


def eval_mcspace_pairwise(respath, gt_data, seeds):
    aucval = eval_mcspace_pairwise_auc(respath, gt_data, seeds)
    f1 = np.nan #eval_mcspace_pairwise_f1(respath, reads, gt_theta)
    aucval_otuthreshold = np.nan #eval_mcspace_pairwise_auc_vary_otu_threshold(respath, reads, gt_theta)
    return aucval, f1, aucval_otuthreshold


def evaluate_case(modelpath, datapath, results, ds, procvar, pertvar, garbvar):
# def evaluate_case(modelpath, datapath, results, ds, npart, nreads, base_sample):
    # case = f"D{ds}_P{npart}_R{nreads}_B{base_sample}"
    # seeds = np.arange(10)
    n_seeds = 1 # TODO: change to 10
    seeds = np.arange(n_seeds)
    case = f"D{ds}_PROC{procvar}_PERT{pertvar}_G{garbvar}"
    dsetname = f"data_D{ds}_timeseries.pkl"

    respath = modelpath / case #/ f"seed_{seed}"

    # load ground truth data 
    gtdata = pickle_load(datapath / dsetname)

    # eval fisher
    aucval, f1, aucval_otuthreshold = eval_mcspace_pairwise(respath, gtdata, seeds)

    results['model'].append('mcspace')
    
    results['process_var_scale'].append(procvar)
    results['perturbation_var_scale'].append(pertvar)
    results['garbage_var_scale'].append(garbvar)

    results['dataset'].append(ds)
    results['auc'].append(aucval)
    return results


def main(rootdir, outdir):
    rootpath = Path(rootdir)
    outpath =  rootpath / "results" / "sensitivity_analysis"  / "pairwise" / "mcspace_results"
    outpath.mkdir(exist_ok=True, parents=True)

    st = time.time()
    
    # result
    results = {}
    results['model'] = []
    # results['base_sample'] = []
    # results['number particles'] = []
    # results['number reads'] = []
    results['process_var_scale'] = []
    results['perturbation_var_scale'] = []
    results['garbage_var_scale'] = []
    results['dataset'] = []
    results['auc'] = []

    #* cases
    process_var_scale = [10, 100, 1000]
    perturbation_var_scale = [100, 1000, 10000]
    garbage_var_scale = [10, 100, 1000]
    dsets = np.arange(2) #TODO: change back10)

    modelpath = Path(outdir) / "sensitivity_analysis" / "assemblage_recovery" / "mcspace"
    datapath = Path(outdir) / "sensitivity_analysis" / "synthetic" # path to semi-synthetic data


    for ds in dsets:
        print(f"analyzing dataset {ds}...")

        # vs process variance
        for procvar in process_var_scale:
            print(f"\tanalyzing process variance={procvar}")
            results = evaluate_case(modelpath, datapath, results, ds, procvar, "default", "default")

        # vs perturbation variance
        for pertvar in perturbation_var_scale:
            print(f"\tanalyzing perturbation variance={pertvar}")
            results = evaluate_case(modelpath, datapath, results, ds, "default", pertvar, "default")
        
        # vs garbage variance
        for garvar in garbage_var_scale:
            print(f"\tanalyzing garbage variance={garvar}")
            results = evaluate_case(modelpath, datapath, results, ds, "default", "default", garvar)

    # save results
    pickle_save(outpath / "results.pkl", results)

    # get the execution time
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
    print("***ALL DONE***")


if __name__ == "__main__":
    # TODO: add back
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-d", dest='rootpath', help='root path')
    # parser.add_argument("-o", dest='outpath', help='output path')
    # args = parser.parse_args()
    # main(args.rootpath, args.outpath)

    rootdir = "./MCSPACE_paper"
    outdir = "./MCSPACE_paper/output/"

    main(rootdir, outdir)
    