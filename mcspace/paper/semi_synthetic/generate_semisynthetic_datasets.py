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


def generate_dataset(beta, theta, num_particles, negbin_n, negbin_p, num_reads=None):
    if num_reads is None:
        counts, z = gen_reads_negbin(beta, theta, num_particles, negbin_n, negbin_p)
    else:
        counts, z = gen_reads_fixed_rdepth(beta, theta, num_particles, num_reads)
    # single time point and subject
    reads = {}
    reads[0] = {}
    reads[0]['s1'] = counts
    assignments = {}
    assignments[0] = {}
    assignments[0]['s1'] = z
    return reads, assignments


def sample_base_dataset(beta, theta, nclust):
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

    newbeta = beta[new_comm_ind]
    # renormalize
    newbeta = np.asarray(newbeta).astype('float64')
    newbeta = newbeta / np.sum(newbeta, axis=0, keepdims=True)
    return newbeta, newtheta


def get_savedata(reads, assignments, theta, beta):
    savedata = {'reads': reads, 
                'assignments': assignments, 
                'theta': theta, 
                'beta': beta}
    return savedata


def gen_semisyn_data(base_sample):
    np.random.seed(42)
    torch.manual_seed(42)

    #* paths
    rootpath = Path("./")
    basepath = rootpath / "paper" / "semi_synthetic"
    datapath = basepath / "base_run" / base_sample

    outpath = basepath / f"semisyn_data" / base_sample
    outpath.mkdir(exist_ok=True, parents=True)

    results = pickle_load(datapath / "results.pkl")
    beta = np.squeeze(results['beta'])
    theta = results['theta']

    print(beta.shape)
    print(theta.shape)

    #* cases
    npart_cases = [10000, 5000, 1000, 500, 100]
    nreads_cases = [10000, 5000, 1000, 500, 100]
    nclust_cases = [5, 10, 15, 20, 25]
    dsets = np.arange(10)

    #* default values
    datafit = pickle_load(basepath / f"negbin_fit_params_{base_sample}_data.pkl")
    npart_default = datafit["num_particles"]
    print(f"default number of particles = {npart_default}")
    negbin_n = datafit['negbin_n']
    negbin_p = datafit['negbin_p']
    print(f"negbin params: p = {negbin_p}, n = {negbin_n}")
    nclust_default = 15


    for ds in dsets:
        print(f"simulating data set {ds}...")

        #* generate data, vary number of clusters
        for nk in nclust_cases:
            newbeta, newtheta = sample_base_dataset(beta, theta, nk)
            reads, assignments = generate_dataset(newbeta, newtheta, npart_default, negbin_n, negbin_p)
            savedata = get_savedata(reads, assignments, newtheta, newbeta)
            pickle_save(outpath / f"data_D{ds}_K{nk}_Pdefault_Rdefault_B{base_sample}.pkl", savedata)

        #* sample new beta and theta
        newbeta, newtheta = sample_base_dataset(beta, theta, nclust_default)

        #* generate samples, vary particles
        npart = npart_cases[0]
        reads, assignments = generate_dataset(newbeta, newtheta, npart, negbin_n, negbin_p)
        savedata = get_savedata(reads, assignments, newtheta, newbeta)
        pickle_save(outpath / f"data_D{ds}_Kdefault_P{npart}_Rdefault_B{base_sample}.pkl", savedata)
        for npart in npart_cases[1:]:
            reads, assignments = down_sample_particles(reads, assignments, npart)
            savedata = get_savedata(reads, assignments, newtheta, newbeta)
            pickle_save(outpath / f"data_D{ds}_Kdefault_P{npart}_Rdefault_B{base_sample}.pkl", savedata)

        #* vary number of reads
        for nreads in nreads_cases:
            reads, assignments = generate_dataset(newbeta, newtheta, npart_default, negbin_n=None, negbin_p=None, num_reads=nreads)
            savedata = get_savedata(reads, assignments, newtheta, newbeta)
            pickle_save(outpath / f"data_D{ds}_Kdefault_Pdefault_R{nreads}_B{base_sample}.pkl", savedata)

    print(f"DONE: {base_sample} semisyn")


def main():
    gen_semisyn_data('Mouse')
    gen_semisyn_data('Human')


if __name__ == "__main__":
    main()
