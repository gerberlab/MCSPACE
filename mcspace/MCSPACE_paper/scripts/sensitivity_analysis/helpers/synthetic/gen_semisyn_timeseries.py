
#! need to add process variance, perturbation prob and magnitude
#! how to get esimate of process variance?...

from pathlib import Path
import numpy as np
from mcspace.utils import pickle_load, pickle_save, get_gt_assoc
from scipy.special import softmax, logsumexp
import torch
from mcspace.model import MCSPACE


def gen_timeseries_beta():
    """
    Generate a synthetic beta vector for timeseries data.
    The beta vector represents the probability of each microbial community being present at each time point.
    """

    #! below is a placeholder for the actual parameters that would be passed in from the model fit
    # TODO: pass in parameters from model fit


    num_communities = 10  # Number of microbial communities
    
    times = [10, 18, 35, 43, 57, 65, 76]
    num_timepoints = len(times)    # Number of time points

    pert_times = [18, 43, 65]  # Perturbation times
    pert_state = [0,1,-1,1,-1,1,-1]

    num_perturbations = len(pert_times)  # Number of perturbations
    num_subjects = 3 # Number of subjects

    #! Key parameters to figure out...
    process_variance = 0.001*np.ones(num_subjects) #* ***seems to work
    perturbation_prob = 0 #0.33
    perturbation_magnitude_mean = 1.0
    perturbation_magnitude_var = 10

    # TODO: check with fixed perturbations; figure out how to get these from model fit

    x_initial = np.random.rand(num_communities,num_subjects) - 0.5 # TODO: get this from model fit
    #! above from model fit

    # initialize the latent variable for the communities
    x_latent = np.zeros((num_communities, num_subjects, num_timepoints))
    x_latent[:, :, 0] = x_initial  # Set the initial state

    #* save perturbations added as well as a check
    perts_added = np.zeros((num_communities, num_subjects, num_timepoints))

# TODO: figure out how to get perturbations from model fit
    #* save perturbations first, will need to subtract after pert time
    for i,t in enumerate(times):
        if t in pert_times:
            # Generate perturbations for each community
            #! pert_indicators = np.random.binomial(n=1, p=perturbation_prob, size=(num_communities,1))
            pert_indicators = np.zeros((num_communities, 1))
            pert_indicators[[0,5],:] = 1  # For testing, set specific communities to have perturbations

            # perturbations_magnitude = np.random.normal(loc=perturbation_magnitude_mean,
            #                                   scale=np.sqrt(perturbation_magnitude_var), 
            #                                   size=(num_communities, num_subjects))
            perturbations_magnitude = np.zeros((num_communities, num_subjects))
            perturbations_magnitude[0,:] = 2.0
            perturbations_magnitude[5,:] = -2.0

            perturbations = perturbations_magnitude * pert_indicators
            perts_added[:, :, i] = perturbations
# TODO: figure out how to get perturbations from model fit


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
            x_latent[:, :, i] = np.random.normal(loc=eta, scale=np.sqrt(process_variance[None,:]*dt))
    
    beta = softmax(x_latent, axis=0)  # Normalize to get probabilities
    return beta, perts_added, x_latent


def main():
    beta, perts, x = gen_timeseries_beta()

    # Save the generated beta, perturbations, and latent variables
    basepath = Path("./")
    scriptpath = basepath / "MCSPACE_paper" / "scripts" / "sensitivity_analysis" / "helpers" / "synthetic"
    outdir = scriptpath / "test_output"
    outdir.mkdir(parents=True, exist_ok=True)
    pickle_save(outdir / "sim_data_test.pkl", {"beta": beta,
                                                "perturbations": perts,
                                                "x_latent": x})
    print("Generated synthetic timeseries beta and saved to test_output/sim_data_test.pkl")


if __name__ == "__main__":
    main()
