from pathlib import Path
import numpy as np
from mcspace.utils import pickle_load, pickle_save, get_gt_assoc
from scipy.special import softmax, logsumexp
import torch
from mcspace.model import MCSPACE


def gen_timeseries_beta(params):
    """
    Generate a synthetic beta vector for timeseries data.
    The beta vector represents the probability of each microbial community being present at each time point.
    """
    beta = params["beta"]  # Load beta from params
    num_communities, num_subjects, num_timepoints = beta.shape
    times = params["times"]  # Load times from params
    pert_times = params["pert_times"]  # Load perturbation times from params
    pert_state = params["pert_state"]  # Load perturbation states from params
    process_variance = params["process_var"]  # Load process variance from params
    perturbation_prob = params["perturbation_probs"]  # Load perturbation probability from params
    perturbation_magnitude_mean = params["perturbation_magnitude_mean"]  # Load perturbation magnitude mean from params
    perturbation_magnitude_var = params["perturbation_magnitude_var"]  # Load perturbation magnitude variance from params

    num_perturbations = len(pert_times)  # Number of perturbations
    num_subjects = 3 # Number of subjects

    # Initial state of the latent variable (invert softmax -> log of beta at time 0)
    x_initial = np.log(beta[:, :, 0])  
    x_initial = x_initial - np.mean(x_initial,axis=0,keepdims=True)

    # initialize the latent variable for the communities
    x_latent = np.zeros((num_communities, num_subjects, num_timepoints))
    x_latent[:, :, 0] = x_initial  # Set the initial state

    #* save perturbations added as well as a check
    perts_added = np.zeros((num_communities, num_subjects, num_timepoints))

    for i,t in enumerate(times):
        if t in pert_times:
            pert_indicators = np.random.binomial(n=1, p=perturbation_prob, size=(num_communities,1))
            perturbations_magnitude = np.random.normal(loc=perturbation_magnitude_mean,
                                              scale=np.sqrt(perturbation_magnitude_var), 
                                              size=(num_communities, 1))
            pmag = np.sign(perturbations_magnitude)*np.log(np.abs(perturbations_magnitude) + 1e-10)  # Avoid log(0)
            perturbations = pmag * pert_indicators
            perts_added[:, :, i] = perturbations

    #* generate latent proportions over time
    for i,t in enumerate(times):
        if i == 0:
            continue
        else:
            # Update the latent variable with perturbations
            # pert 
            if pert_state[i] == 1:
                # perturbation turns on
                eta = x_latent[:, :, i-1] + perts_added[:, :, i]
            elif pert_state[i] == -1:
                # perturbation turns off
                #! note, this does not account for perturbations that are still on, where we would just have drift between time points...
                # TODO: indictors should actually loop over p, not t
                eta = x_latent[:, :, i-1] - perts_added[:, :, i-1] 
            elif pert_state[i] == 0:
                # no perturbation
                eta = x_latent[:, :, i-1]
            else:
                raise ValueError("pert_state must be 1, 0, or -1")

            # add process noise
            dt = times[i] - times[i-1]  # Time difference between current and previous time point
            x_latent[:, :, i] = np.random.normal(loc=eta, scale=np.sqrt(0.1*process_variance[None,:]*dt))
    
    beta = softmax(x_latent, axis=0)  # Normalize to get probabilities
    return beta, perts_added, x_latent


def sample_base_dataset(params):
    theta = params['theta'].T  # Load theta from params
    beta = params['beta']  # Load beta from params

    nclust, nsubj, ntime = beta.shape

    #* generate new theta (assemblages)
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

    #* generate new beta (proportions)
    newbeta, perts_added, x_latent = gen_timeseries_beta(params)


    # # renormalize
    # newbeta = np.asarray(newbeta).astype('float64')
    # newbeta = newbeta / np.sum(newbeta, axis=0, keepdims=True)
    return newbeta, newtheta


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


def generate_dataset(beta, theta, times, subjects, num_particles, negbin_n, negbin_p, num_reads=None, pi_garb=None):
    nsubj = len(subjects)
    reads = {}
    assignments = {}
    gclusts = {}

    for i,t in enumerate(times):
        gclust = (1.0/nsubj)*(beta[:,:,i,None]*theta[:,None,:]).sum(axis=(0,1))
        theta_mix = (1.0 - pi_garb)*theta + pi_garb*gclust[None,:] 
        reads[t] = {}
        assignments[t] = {}
        gclusts[t] = gclust
        for j,s in enumerate(subjects):
            if num_reads is None:
                counts, z = gen_reads_negbin(beta[:,j,i], theta_mix, num_particles, negbin_n, negbin_p)
            else:
                counts, z = gen_reads_fixed_rdepth(beta[:,j,i], theta_mix, num_particles, num_reads)
            reads[t][s] = counts
            assignments[t][s] = z    
    return reads, assignments, gclusts



def get_negbin_mean(n,p):
    return n*(1.0-p)/p


def get_negbin_var(n,p):
    return n*(1.0-p)/(p**2)


def get_negbin_alpha(mu, var):
    return (var-mu)/(mu**2)


def get_negbin_var_from_alpha(mu, alpha):
    return mu + alpha*(mu**2)


def get_negbin_p(mu, alpha):
    var = get_negbin_var_from_alpha(mu, alpha)
    return mu/var


def get_negbin_n(mu, alpha):
    var = get_negbin_var_from_alpha(mu, alpha)
    return (mu**2)/(var - mu)


def main(rootdir, outdir):
    np.random.seed(42)
    torch.manual_seed(42)

    #* paths
    rootpath = Path(rootdir)
    basepath = rootpath / "scripts" / "sensitivity_analysis" / "helpers" / "synthetic"

    outpath = Path(outdir) / "sensitivity_analysis" / "synthetic"
    outpath.mkdir(exist_ok=True, parents=True)

    params = pickle_load(basepath / "time_series_params.pkl")
    theta = params['theta']  # Load theta from params

    dsets = np.arange(10)

    #* default values
    datafits = pickle_load(basepath / "negbin_fit_params_Mouse_data.pkl")

    negbin_p = datafits['negbin_p']
    negbin_n = datafits['negbin_n']
    num_particles = int(datafits['num_particles'])

    pi_garb = params['pi_garb']
    times = params['times']
    subjects = params['subjects']
    perturbed_times = [0, 1, -1, 1, -1, 1, -1]


    print(f"negbin params: p = {negbin_p}, n = {negbin_n}")

    mu = get_negbin_mean(negbin_n, negbin_p)
    var = get_negbin_var(negbin_n, negbin_p)
    negbin_alpha = get_negbin_alpha(mu, var)
    print(f"negbin params: p = {negbin_p}, n = {negbin_n}; mu = {mu}, var = {var}, alpha = {negbin_alpha}")

    # SIMULATE DATASETS
    for ds in dsets:
        print(f"Generating dataset {ds+1} of {len(dsets)}")

        #* sample new beta and theta
        newbeta, newtheta = sample_base_dataset(params)

        reads, assignments, gclust = generate_dataset(newbeta, newtheta, times, subjects, num_particles, negbin_n, negbin_p, pi_garb=pi_garb)
        savedata = {'reads': reads,
                    'assignments': assignments,
                    'theta': newtheta,
                    'beta': newbeta,
                    'gclust': gclust,
                    'pi_garb': pi_garb,
                    'times': times,
                    'subjects': subjects,
                    'perturbed_times': perturbed_times}
        pickle_save(outpath / f"data_D{ds}_timeseries.pkl", savedata)
    print("\n\n***All datasets generated successfully!***\n\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", dest='rootdir', help='project directory path')
    parser.add_argument("-o", dest="outdir", help="output directory path")
    args = parser.parse_args()
    main(args.rootdir, args.outdir)
