import os
os.environ["OMP_NUM_THREADS"] = "40" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "40" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "40" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "40" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "40" # export NUMEXPR_NUM_THREADS=6

import torch
import numpy as np
from pathlib import Path
from mcspace.utils import pickle_load, pickle_save, RESULT_FILE, MODEL_FILE, hellinger_distance
from mcspace.data_utils import get_data
from mcspace.utils import ilr_transform_data, inv_ilr_transform_data, flatten_data, get_device
from sklearn.metrics import normalized_mutual_info_score as calcNMI
import time
import os
"""
evaluate NMI and community reconstruction error for mcspace, gmm, gmm-1d, and gmm-2d
"""

# TODO: add number of cluster error...


def sample_assignments_subj(otu_dist, comm_dist, counts):
    EPS = 1e-10
    num_particles, num_communities, num_otus = otu_dist.shape

    #* compute posterior assignment probabilities
    logprob = (counts[:,None,:]*np.log(otu_dist + EPS)).sum(axis=-1) + np.log(comm_dist + EPS)
    logprob[comm_dist < EPS] = -np.inf

    #* sample from categorical
    g = np.random.gumbel(size=(num_particles, num_communities))
    z = np.argmax(g + logprob, axis=1)
    return z


# def get_assignment_samples(model, data, flatcounts, subjlabels, n_samples):
#     # TODO:...
#     #! DOUBLE CHECK; clean up code...
#     # get flattened array of samples...
#     npart, notus = flatcounts.shape
#     # ngrps = len(contam_comm_dict.keys())
#     # contam_comm = np.zeros((ngrps, notus))
#     # for i,grp in enumerate(contam_comm_dict.keys()):
#     #     contam_comm[i,:] = contam_comm_dict[grp].cpu().detach().clone().numpy()

#     zsamp = np.zeros((n_samples, npart))
#     #* take samples and compute assignments for each particle
#     for i in range(n_samples):
#         print(f"sample {i} of {n_samples}")
#         _, theta, beta, gamma = model(data)
#         theta = theta.cpu().detach().clone().numpy()
#         beta = beta.cpu().detach().clone().numpy()
#         # pi_garb = pi_garb.cpu().detach().clone().numpy()

#         # dealing with dict; using flattened labels
#         exp_beta = beta[:,subjlabels[:,0],subjlabels[:,1]].T
#         # cw = pi_garb[subjlabels[:,0]]
#         # ccomm = contam_comm[subjlabels[:,0]]
#         exp_theta = theta[None,:,:] #(1.0-cw[:,None,None])*theta[None,:,:] + (cw[:,None,None])*ccomm[:,None,:]
#         tassign = sample_assignments_subj(exp_theta, exp_beta, flatcounts)
#         zsamp[i,:] = tassign
#     return zsamp


def get_assignment_samples(model, data, flatcounts, subjlabels, n_samples):
    # TODO:...
    #! DOUBLE CHECK; clean up code...
    # get flattened array of samples...
    npart, notus = flatcounts.shape
    contam_comm_dict = data['group_garbage_clusters']
    ngrps = len(contam_comm_dict.keys())
    contam_comm = np.zeros((ngrps, notus))
    for i,grp in enumerate(contam_comm_dict.keys()):
        contam_comm[i,:] = contam_comm_dict[grp].cpu().detach().clone().numpy()

    zsamp = np.zeros((n_samples, npart))
    #* take samples and compute assignments for each particle
    for i in range(n_samples):
        print(f"sample {i} of {n_samples}")
        _, theta, beta, gamma, pi_garb = model(data)
        theta = theta.cpu().detach().clone().numpy()
        beta = beta.cpu().detach().clone().numpy()
        pi_garb = pi_garb.cpu().detach().clone().numpy()

        # dealing with dict; using flattened labels
        exp_beta = beta[:,subjlabels[:,0],subjlabels[:,1]].T
        cw = pi_garb[subjlabels[:,0]]
        ccomm = contam_comm[subjlabels[:,0],:]
        # exp_theta = theta[None,:,:] #(1.0-cw[:,None,None])*theta[None,:,:] + (cw[:,None,None])*ccomm[:,None,:]
        exp_theta = (1.0-cw[:,None,None])*theta[None,:,:] + (cw[:,None,None])*ccomm[:,None,:]
        tassign = sample_assignments_subj(exp_theta, exp_beta, flatcounts)
        zsamp[i,:] = tassign
    return zsamp


def calc_nmi_samples(z_samples, gtassign):
    nsamp = z_samples.shape[0]
    nmi_scores = np.zeros(nsamp)
    for i in range(nsamp):
        nmi_scores[i] = calcNMI(gtassign, z_samples[i,:])
    return nmi_scores


def calc_nearest_comms_error(learned_theta, gt_theta):
    # use greedy algo to get closest mapping
    # take error in both directions and take average
    
    # calcuate a distance matrix
    nk_learned = learned_theta.shape[0]
    nk_gt = gt_theta.shape[0]
    distmat = np.zeros((nk_learned, nk_gt))
    for i in range(nk_learned):
        for j in range(nk_gt):
            distmat[i,j] = hellinger_distance(learned_theta[i,:], gt_theta[j,:])

    niter = min(distmat.shape)
    ntotal = max(distmat.shape)

    roldist = 0
    for _ in range(niter):
        ind = np.unravel_index(np.argmin(distmat, axis=None), distmat.shape)
        minval = distmat[ind]
        # print(minval)
        roldist += minval
        # remove row and column from matrix
        rows_keep = [i for i in range(distmat.shape[0]) if i != ind[0]]
        cols_keep = [j for j in range(distmat.shape[1]) if j != ind[1]]
        distmat = distmat[rows_keep,:][:,cols_keep]
        # print(distmat.shape)
    # add ones for each dim mismatch
    roldist += (ntotal-niter)
    roldist /= ntotal
    return roldist


def get_mcspace_comm_errors(model, data, gt_theta, n_samples):
    errors = np.zeros(n_samples)
    nk_learned = np.zeros(n_samples)
    for i in range(n_samples):
        model.train()
        _, sample_theta, sample_beta, gamma, _ = model(data)
        sample_beta = sample_beta.cpu().detach().clone().numpy()
        subset = sample_beta.sum(axis=(1,2))>1e-10
        sample_theta = sample_theta.cpu().detach().clone().numpy()
        # take subset of sampled communities
        sample_theta = sample_theta[subset,:]
        sample_error = calc_nearest_comms_error(sample_theta, gt_theta)
        errors[i] = sample_error
        nk_learned[i] = gamma.cpu().detach().clone().numpy().sum() + 1
    return errors, nk_learned


def get_mcspace_min_loss_seed(respath, gt_data, seeds):
    device = get_device()
    reads = gt_data['reads']
    data = get_data(reads, device)

    n_samples = 100
    losses = {}
    for seed in seeds:
        modelpath = respath / f"seed_{seed}"
        model = torch.load(modelpath / MODEL_FILE)
    
        loss_samples = np.zeros(n_samples)
        for i in range(n_samples):
            loss, _, _, _, _ = model(data)
            loss_samples[i] = loss.item()
        losses[seed] = np.mean(loss_samples)
    
    best_seed = min(losses, key=losses.get)
    return best_seed


def eval_mcspace_metrics(respath, gt_data, seeds):
    device = get_device()

    # min_seed = seeds[0] #! get min loss seed...
    min_seed = get_mcspace_min_loss_seed(respath, gt_data, seeds)
    modelpath = respath / f"seed_{min_seed}"

    # take posterior samples and evaluate nmi and comm error for each
    model = torch.load(modelpath / MODEL_FILE)
    reads = gt_data['reads']
    data = get_data(reads, device)

    n_samples = 100
    comm_err_samples, nk_learned = get_mcspace_comm_errors(model, data, gt_data['theta'], n_samples)

    flatreads, subj_labels, cluster_labels = flatten_data(gt_data)
    z_samples = get_assignment_samples(model, data, flatreads, subj_labels, n_samples)
    nmi_samples = calc_nmi_samples(z_samples, cluster_labels)

    # take median results and return
    nmi = np.median(nmi_samples)
    comm_err = np.median(comm_err_samples)
    return nmi, comm_err, nk_learned


def get_best_gmm_model(casepath, nk_list, seed_list):
    # loop over seeds and k, compute average aic for each k
    # take k with min aic and seed with min aic given k
    n_k = len(nk_list)
    n_seed = len(seed_list)
    aics = np.zeros((n_k, n_seed))

    for i, k in enumerate(nk_list):
        for j, seed in enumerate(seed_list):
            respath = casepath / f"K_{k}_seed_{seed}"
            params = pickle_load(respath / RESULT_FILE)
            aics[i,j] = params['aic']
    
    aicmed = np.median(aics, axis=1)
    min_idx = np.argmin(aicmed)
    kmin = nk_list[min_idx]
    min_seed_idx = np.argmin(aics[min_idx,:])
    seedmin = seed_list[min_seed_idx]
    return kmin, seedmin


def eval_gmm_metrics(respath, gt_data, klist, seeds):
    # find k,seed with min AIC
    kmin, seedmin = get_best_gmm_model(respath, klist, seeds)
    modelpath = respath / f"K_{kmin}_seed_{seedmin}"
    #* get community error
    res = pickle_load(modelpath / RESULT_FILE)
    comm_err = calc_nearest_comms_error(res['theta'], gt_data['theta'])
    #* calc nmi
    learned_labels = res['labels']
    flatreads, subj_labels, cluster_labels = flatten_data(gt_data)
    nmi = calcNMI(cluster_labels, learned_labels)
    return nmi, comm_err, kmin


def evaluate_case(basepath, datapath, res, ds, nk, npart, nreads, gweight, nsubj, base_sample):
    seeds = np.arange(5)

    case = f"D{ds}_K{nk}_P{npart}_R{nreads}_G{gweight}_B{base_sample}_S{nsubj}"

    # load ground truth data 
    gtdata = pickle_load(datapath / f"data_{case}.pkl")

    # evaluate model
    models = ["mcspace", "gmm_basic"]
    ncomm_lists = ["pass", np.arange(2,51)]
    for mod, klist in zip(models, ncomm_lists):
        respath = basepath / f"{mod}_runs" / base_sample / case
        if mod == "mcspace":
            nmi, comm_err, nk_learned_samples = eval_mcspace_metrics(respath, gtdata, seeds)
            # TODO:
            nk_learned = np.median(nk_learned_samples)
            # nk_error = np.median(nk_error_samples)
        else:
            nmi, comm_err, nk_learned = eval_gmm_metrics(respath, gtdata, klist, seeds)
            # if nk == "default":
            #     nk_error = np.abs(23 - nk_learned)
            # else:
            #     nk_error = np.abs(nk - nk_learned)

        res['model'].append(mod)
        res['base_sample'].append(base_sample)
        res['number particles'].append(npart)
        res['read depth'].append(nreads)
        res['number clusters'].append(nk)
        res['contamination weight'].append(gweight)
        res['number subjects'].append(nsubj)
        res['dataset'].append(ds)
        res['NMI'].append(nmi)
        res['community error'].append(comm_err)
        res['number clusters learned'].append(nk_learned)
    return res


def main(rootdir):
    rootpath = Path(rootdir)
    # rootpath = Path("/PHShome/gu021/mcspace_paper/mcspace")
    basepath = rootpath / "paper_cluster" / "assemblage_recovery"
    
    st = time.time()

    # loop over cases; save results as dataframe here
    results = {}
    results['model'] = []
    results['base_sample'] = []
    results['number particles'] = []
    results['read depth'] = []
    results['number clusters'] = []
    results['contamination weight'] = []
    results['number subjects'] = []
    results['dataset'] = []
    results['NMI'] = []
    results['community error'] = []
    results['number clusters learned'] = []


    for base_sample in ['Mouse', 'Human']:
        #* cases
        if base_sample == 'Mouse':
            nsubj_cases = [1,3,5,7,10]
        npart_cases = [10000, 5000, 1000, 500, 100]
        nreads_cases = [10000, 5000, 1000, 500, 100]
        nclust_cases = [5, 10, 15, 20, 25]
        pgarb_cases = [0.0, 0.025, 0.05, 0.075, 0.1]
        dsets = np.arange(10)

        datapath = rootpath / "paper_cluster" / "semi_synthetic_data" / "semisyn_data" / base_sample
        for ds in dsets:
            print(f"\n\n...analyzing dataset {ds}")
            # vs number clusters
            print("nclust cases")
            for nk in nclust_cases:
                results = evaluate_case(basepath, datapath, results, ds, nk, "default", "default", "default", "default", base_sample)
                print(f"\t {nk}")

            # vs number particles-
            print("npart cases")
            for npart in npart_cases:
                results = evaluate_case(basepath, datapath, results, ds, "default", npart, "default", "default", "default", base_sample)
                print(f"\t {npart}")

            # vs read depth
            print("nreads cases")
            for nreads in nreads_cases:
                results = evaluate_case(basepath, datapath, results, ds, "default", "default", nreads, "default", "default", base_sample)
                print(f"\t {nreads}")    

            # vs contamination weight
            print("contamination cases")
            for gweight in pgarb_cases:
                results = evaluate_case(basepath, datapath, results, ds, "default", "default", "default", gweight, "default", base_sample)
                print(f"\t {gweight}")    

            # vs subject
            if base_sample == "Mouse":
                print("subject cases")
                for nsubj in nsubj_cases:
                    results = evaluate_case(basepath, datapath, results, ds, "default", "default", "default", "default", nsubj, base_sample)
                    print(f"\t {nsubj}")    
    
    # save results
    outpath = basepath / "eval_results"
    outpath.mkdir(exist_ok=True, parents=True)
    pickle_save(outpath / "results.pkl", results)

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

