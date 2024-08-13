import os
os.environ["OMP_NUM_THREADS"] = "40" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "40" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "40" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "40" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "40" # export NUMEXPR_NUM_THREADS=6

import numpy as np
import torch
from pathlib import Path
from mcspace.utils import pickle_load, pickle_save, sample_assignments, sample_reads, \
    get_cosine_dist, RESULT_FILE, MODEL_FILE, get_device
# from mcspace.models import BasicModel
from mcspace.data_utils import get_data, get_human_timeseries_dataset, get_mouse_diet_perturbations_dataset
import time





#* MCSPACE methods =======================================================
def eval_cos_err_mcspace_samples(model, data, testcounts, dsreads, n_samples):
    n_holdout, _ = testcounts.shape
    cos_err = np.zeros((n_samples,n_holdout))

    test_ra = testcounts/testcounts.sum(axis=1, keepdims=True)
    read_depths = testcounts.sum(axis=1)
    for i in range(n_samples):
        print(f"sample {i} of {n_samples}")
        # _, theta, beta, pi_garb, gamma = model(data)
        _, theta, beta, gamma = model(data)
        theta = theta.cpu().detach().clone().numpy()
        beta = beta[:,0,0].cpu().detach().clone().numpy() #! single time and subject 
        tassign = sample_assignments(theta, beta, dsreads)
        s_reads = sample_reads(theta, tassign, read_depths)
        cos_err[i,:] = get_cosine_dist(testcounts, s_reads)
    return cos_err, read_depths


def evaluate_mcspace(case, fold, basepath, datapath):
    def _convert_data_to_dict(reads):
        counts = {}
        counts[0] = {}
        counts[0]['s1'] = reads
        return counts

    device = get_device()
    modelpath = basepath / "mcspace" / case / f"Fold_F{fold}"
    model = torch.load(modelpath / MODEL_FILE)
    #* load train data to sample from posterior
    traindata = pickle_load(datapath / f"train_F{fold}.pkl")
    traindata = _convert_data_to_dict(traindata)
    traindata = get_data(traindata, device)
    #* load test data
    testfile = datapath / f"test_F{fold}.pkl"
    testdata = pickle_load(testfile)
    #* get downsampled reads
    dsreads = pickle_load(datapath / f"ds0.5_F{fold}.pkl")
    #* for each heldout particle, get predictions and evaluate metrics
    n_samples = 100
    cos_err, read_depths = eval_cos_err_mcspace_samples(model, traindata, testdata, dsreads, n_samples)
    cos_err_med = np.median(cos_err, axis=0)

    return cos_err_med, read_depths


#* GMM methods =======================================================
def get_read_counts(theta, assigns, read_depths):
    ncomm, notus = theta.shape
    npart = len(assigns)
    reads = np.zeros((npart, notus))
    for lidx in range(npart):
        dist = theta[assigns[lidx],:]
        dist = np.squeeze(np.asarray(dist).astype('float64'))
        dist = dist / np.sum(dist)
        rd = read_depths[lidx]
        rsamp = dist*rd
        reads[lidx,:] = rsamp
    return reads


def eval_cos_err_gmm(model, testcounts, dsreads):
    cos_err_med_all = []
    read_depths_all = []

    test_ra = testcounts/testcounts.sum(axis=1, keepdims=True)
    read_depths = testcounts.sum(axis=1)
    tassign = model.predict_labels(dsreads)
    theta = model.get_communities()
    s_reads = get_read_counts(theta, tassign, read_depths)
    cos_err = get_cosine_dist(testcounts, s_reads)

    return cos_err, read_depths


def get_min_aic_k(modelbasepath, klist):
    aics = {}
    for k in klist:
        res = pickle_load(modelbasepath / f"K_{k}" / RESULT_FILE)
        aics[k] = res['aic']
    min_aic_k = min(aics, key=aics.get)
    return min_aic_k


def evaluate_gmm(case, fold, basepath, datapath, modeldir, klist):
    modelbasepath = basepath / modeldir / case / f"Fold_F{fold}"
    min_aic_k = get_min_aic_k(modelbasepath, klist)
    modelpath = modelbasepath / f"K_{min_aic_k}"
    model = pickle_load(modelpath / MODEL_FILE)
    #* load test data
    testfile = datapath / f"test_F{fold}.pkl"
    testdata = pickle_load(testfile)
    #* get downsampled reads
    dsreads = pickle_load(datapath / f"ds0.5_F{fold}.pkl")
    #* for each heldout particle, get predictions and evaluate metrics
    cos_err, read_depths = eval_cos_err_gmm(model, testdata, dsreads)
    return cos_err, read_depths


def evaluate_gmm_basic(case, fold, basepath, datapath):
    modeldir = "gmm_basic"
    klist = np.arange(2,31)
    cos_err, read_depths = evaluate_gmm(case, fold, basepath, datapath, modeldir, klist)
    return cos_err, read_depths


def evaluate_gmm_one_dim(case, fold, basepath, datapath):
    modeldir = "gmm_one_dim"
    klist = np.arange(2,11)
    cos_err, read_depths = evaluate_gmm(case, fold, basepath, datapath, modeldir, klist)
    return cos_err, read_depths


def evaluate_gmm_two_dim(case, fold, basepath, datapath):
    modeldir = "gmm_two_dim"
    klist = [4,6,8,10]
    cos_err, read_depths = evaluate_gmm(case, fold, basepath, datapath, modeldir, klist)
    return cos_err, read_depths


#* Evaluate all cases ========================================================
def evaluate_case(results, case, fold, basepath):
    datapath = basepath / "holdout_data" / case
    methods = [evaluate_mcspace, evaluate_gmm_basic, evaluate_gmm_one_dim, evaluate_gmm_two_dim]
    models = ['mcspace', 'gmm_basic', 'gmm_one_dim', 'gmm_two_dim']
    for mod, met in zip(models, methods):
        cos_err, read_depths = met(case, fold, basepath, datapath)
        for lidx in range(len(cos_err)):
            results['model'].append(mod)
            results['case'].append(case)
            results['fold'].append(fold)
            results['read depth'].append(read_depths[lidx])
            results['particle id'].append(lidx)
            results['cosine distance'].append(cos_err[lidx])
    return results


def get_cases():
    all_cases = []

    dsets = [get_human_timeseries_dataset, get_mouse_diet_perturbations_dataset]
    names = ['Human', 'Mouse']

    for dset, name in zip(dsets, names):
        reads, num_otus, times, subjects, dataset = dset()
        for t in times:
            for s in subjects:
                case = f"{name}_{t}_{s}"
                all_cases.append(case)
    return all_cases


def main():
    st = time.time()
    #* cases
    nfolds = 5

    rootpath = Path("./")
    basepath = rootpath / "paper_cluster" / "cross_validation"

    results = {}
    results['model'] = []
    results['case'] = []
    results['fold'] = []
    results['read depth'] = []
    results['particle id'] = []
    results['cosine distance'] = []

    # loop over cases
    cases = get_cases()
    for case in cases:
        for fold in range(nfolds):
            # eval cosine error for models
            results = evaluate_case(results, case, fold, basepath)
            print(f"DONE |  fold={fold}")
        print(f"DONE |  case={case}")

    # save results
    outpath = basepath / "cv_holdout_results"
    outpath.mkdir(exist_ok=True, parents=True)
    pickle_save(outpath / "results.pkl", results)
    # get the execution time
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
    print("***ALL DONE***")


if __name__ == "__main__":
    main()
