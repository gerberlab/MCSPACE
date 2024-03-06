from pathlib import Path
import numpy as np
from mcspace.utils import pickle_load, pickle_save, MODEL_FILE, DATA_FILE, get_summary_stats
from scipy.special import softmax, logsumexp
import torch
from mcspace.model import MCSPACE

#! need ot update methods w/dict or not
#! left off here...

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
        # TODO: do rejection sampling instead?
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


# vary number of particles:
# 100, 250, 500, 1000, 2500, 5000, 10000

# vary number reads
# 100, 250, 500, 1000, 2500, 5000, 10000

#* realistic values based on real data ^
# 7 for each case; 14 total, for 10 datasets x2 (mouse and human)


def gen_human_semisyn_data(): #TODO: use same for both datasets, input arg difference human vs mouse
    np.random.seed(42)
    torch.manual_seed(0)
    
    # paths
    rootpath = Path("./")
    basepath = rootpath / "paper" / "semi_synthetic"
    datapath = basepath / "base_run" / "Human"

    model = torch.load()
