from pathlib import Path
import numpy as np
from mcspace.utils import pickle_load, pickle_save, get_gt_assoc
from scipy.special import softmax, logsumexp
import torch
from mcspace.model import MCSPACE


def gen_reads_negbin(beta, theta, num_particles, negbin_n, negbin_p):
    """
    The gen_reads function generates a matrix of simulated reads given a set of microbial community compositions (represented by the matrix theta) and the probabilities of each community being present (represented by the array beta).

    Args:
    beta: An array of probabilities representing the probability of each microbial community being present. The length of the array is the number of microbial communities.
    theta: A matrix representing the microbial community compositions. The rows represent the different microbial communities and the columns represent the different OTUs (operational taxonomic units). The shape of the matrix is (num_communities, num_otus).
    num_particles: The number of particles (i.e., samples) to generate reads for.
    read_depth: The read depth for each particle.

    Returns:
       A matrix of simulated reads. The rows represent the different particles and the columns represent the different OTUs. The shape of the matrix is (num_particles, num_otus).
    """
    num_communities, num_otus = theta.shape
    reads = np.zeros((num_particles, num_otus))
    z = np.random.choice(num_communities, p=beta, size=num_particles)
    for lidx in range(num_particles):
        # sample read depth from negbin
        read_depth = max(np.random.negative_binomial(negbin_n, negbin_p), 10) # MINIMUM OF 10 READS
        reads[lidx,:] = np.random.multinomial(n=read_depth, pvals=theta[z[lidx],:])
    return reads, z


def gen_reads_fixed_rdepth(beta, theta, num_particles, read_depth):
    num_communities, num_otus = theta.shape
    reads = np.zeros((num_particles, num_otus))
    z = np.random.choice(num_communities, p=beta, size=num_particles)
    for lidx in range(num_particles):
        reads[lidx,:] = np.random.multinomial(n=read_depth, pvals=theta[z[lidx],:])
    return reads, z


def down_sample_particles(full_reads, full_assign, npart):
    groups = list(full_reads.keys())
    subjs = list(full_reads[groups[0]].keys())
    reads = {}
    assign = {}
    for grp in groups:
        reads[grp] = {}
        assign[grp] = {}
        for sub in subjs:
            tmp = full_reads[grp][sub].copy()
            full_npart = tmp.shape[0]
            psubset = np.random.choice(np.arange(full_npart), size=npart, replace=False)
            reads[grp][sub] = tmp[psubset,:]
            assign[grp][sub] = full_assign[grp][sub][psubset].copy()
    return reads, assign


def generate_dataset(beta, theta, num_particles, negbin_n, negbin_p, num_reads=None, pi_garb=None, gclust=None):
    theta_mix = (1.0 - pi_garb)*theta + pi_garb*gclust[None,:]
    reads = {}
    reads[0] = {}
    assignments = {}
    assignments[0] = {}
    nsubj = beta.shape[1]
    for s in range(nsubj):
        if num_reads is None:
            counts, z = gen_reads_negbin(beta[:,s], theta_mix, num_particles, negbin_n, negbin_p)
        else:
            counts, z = gen_reads_fixed_rdepth(beta[:,s], theta_mix, num_particles, num_reads)
        reads[0][f's{s+1}'] = counts
        assignments[0][f's{s+1}'] = z
    return reads, assignments


def sample_base_dataset(betamean, theta, nclust, nsubj, bvar):
    K = nclust
    Kidx_max, n_otus = theta.shape
    success = False
    while success is False:
        new_comm_ind = np.random.choice(Kidx_max, size=(K,), replace=True)
        newtheta = theta[new_comm_ind,:]

        # permute OTUs in theta
        for kidx in range(K):
            new_order = np.random.permutation(n_otus)
            newtheta[kidx,:] = newtheta[kidx,new_order]
        # renomalize -- shouldn't make much difference
        newtheta = np.asarray(newtheta).astype('float64')
        newtheta = newtheta / np.sum(newtheta, axis=1, keepdims=True)

        # check ground truth associations, retry if everything associated
        gt_assoc = get_gt_assoc(newtheta, otu_threshold=0.005)
        notus = gt_assoc.shape[0]
        triassoc = gt_assoc[np.triu_indices(notus, k=1)]
        proportion_unit = triassoc.sum()/len(triassoc)
        if proportion_unit < 1.0:
            success = True
            print("...success")
        else:
            print("...retry")

    if nsubj > 1:
        x_beta = betamean[new_comm_ind]
        x_subj = np.random.normal(loc=x_beta, scale=np.sqrt(bvar), size=(nsubj,K)).T
        newbeta = softmax(x_subj, axis=0)
    else:
        newbeta = softmax(betamean[new_comm_ind], axis=0)[:,None]
    # renormalize
    newbeta = np.asarray(newbeta).astype('float64')
    newbeta = newbeta / np.sum(newbeta, axis=0, keepdims=True)
    return newbeta, newtheta


def get_savedata(reads, assignments, theta, beta, pigarb):
    savedata = {'reads': reads, 
                'assignments': assignments, 
                'theta': theta, 
                'beta': beta,
                'pi_garbage': pigarb}
    return savedata


def gen_semisyn_data(base_sample):
    np.random.seed(42)
    torch.manual_seed(42)

    #* paths
    rootpath = Path("./")
    basepath = rootpath / "paper_cluster" / "semi_synthetic_data"
    datapath = basepath / "base_run" / base_sample

    outpath = basepath / f"semisyn_data" / base_sample
    outpath.mkdir(exist_ok=True, parents=True)

    results = pickle_load(datapath / "results.pkl")
    beta = np.squeeze(results['beta'])
    theta = results['theta']
    gclust = results['garbage_cluster']

    print(beta.shape)
    print(theta.shape)

    #* cases
    if base_sample == 'Human':
        betamean = np.log(beta)
        betavar = None
        nsubj_default = 1
    else:
        nsubj_cases = [1,3,5,7,10]
        nsubj_default = 3
        logbeta = np.log(beta)
        betamean = np.mean(logbeta, axis=1)
        betavar = np.median(np.var(logbeta, axis=1))

    npart_cases = [10000, 5000, 1000, 500, 100]
    nreads_cases = [10000, 5000, 1000, 500, 100]
    nclust_cases = [5, 10, 15, 20, 25]
    pgarb_cases = [0.0, 0.025, 0.05, 0.075, 0.1]
    dsets = np.arange(10)

    #* default values
    pigarb_default = results['pi_garb']
    datafit = pickle_load(basepath / f"negbin_fit_params_{base_sample}_data.pkl")
    npart_default = datafit["num_particles"]
    print(f"default number of particles = {npart_default}")
    negbin_n = datafit['negbin_n']
    negbin_p = datafit['negbin_p']
    print(f"negbin params: p = {negbin_p}, n = {negbin_n}")
    nclust_default = 15
    

    for ds in dsets:
        print(f"simulating data set {ds}...")

        #* vary number of subjects
        if base_sample == 'Mouse':
            for nsubj in nsubj_cases:
                newbeta, newtheta = sample_base_dataset(betamean, theta, nclust_default, nsubj, betavar)
                reads, assignments = generate_dataset(newbeta, newtheta, npart_default, negbin_n, negbin_p, pi_garb=pigarb_default, gclust=gclust)
                savedata = get_savedata(reads, assignments, newtheta, newbeta, pigarb_default)
                pickle_save(outpath / f"data_D{ds}_Kdefault_Pdefault_Rdefault_Gdefault_B{base_sample}_S{nsubj}.pkl", savedata)

        #* generate data, vary number of clusters
        for nk in nclust_cases:
            newbeta, newtheta = sample_base_dataset(betamean, theta, nk, nsubj_default, betavar)
            reads, assignments = generate_dataset(newbeta, newtheta, npart_default, negbin_n, negbin_p, pi_garb=pigarb_default, gclust=gclust)
            savedata = get_savedata(reads, assignments, newtheta, newbeta, pigarb_default)
            pickle_save(outpath / f"data_D{ds}_K{nk}_Pdefault_Rdefault_Gdefault_B{base_sample}_Sdefault.pkl", savedata)

        #* sample new beta and theta
        newbeta, newtheta = sample_base_dataset(betamean, theta, nclust_default, nsubj_default, betavar)

        #* generate samples, vary particles
        npart = npart_cases[0]
        reads, assignments = generate_dataset(newbeta, newtheta, npart, negbin_n, negbin_p, pi_garb=pigarb_default, gclust=gclust)
        savedata = get_savedata(reads, assignments, newtheta, newbeta, pigarb_default)
        pickle_save(outpath / f"data_D{ds}_Kdefault_P{npart}_Rdefault_Gdefault_B{base_sample}_Sdefault.pkl", savedata)
        for npart in npart_cases[1:]:
            reads, assignments = down_sample_particles(reads, assignments, npart)
            savedata = get_savedata(reads, assignments, newtheta, newbeta, pigarb_default)
            pickle_save(outpath / f"data_D{ds}_Kdefault_P{npart}_Rdefault_Gdefault_B{base_sample}_Sdefault.pkl", savedata)

        #* vary number of reads
        for nreads in nreads_cases:
            reads, assignments = generate_dataset(newbeta, newtheta, npart_default, negbin_n=None, negbin_p=None, num_reads=nreads, pi_garb=pigarb_default, gclust=gclust)
            savedata = get_savedata(reads, assignments, newtheta, newbeta, pigarb_default)
            pickle_save(outpath / f"data_D{ds}_Kdefault_Pdefault_R{nreads}_Gdefault_B{base_sample}_Sdefault.pkl", savedata)

        #* vary contamination weight
        for gpi in pgarb_cases:
            reads, assignments = generate_dataset(newbeta, newtheta, npart_default, negbin_n, negbin_p, pi_garb=gpi, gclust=gclust)
            savedata = get_savedata(reads, assignments, newtheta, newbeta, gpi)
            pickle_save(outpath / f"data_D{ds}_Kdefault_Pdefault_Rdefault_G{gpi}_B{base_sample}_Sdefault.pkl", savedata)


    print(f"DONE: {base_sample} semisyn")


def main():
    gen_semisyn_data('Mouse')
    gen_semisyn_data('Human')


if __name__ == "__main__":
    main()
