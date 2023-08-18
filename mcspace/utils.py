import numpy as np 
import pandas as pd 
import torch
import torch.nn.functional as F
import pickle 
from scipy.stats import spearmanr #, hypergeom
from scipy.special import softmax, logsumexp


MODEL_FILE = "model.pt"
RESULT_FILE = "results.pkl"
DATA_FILE = "data.pkl"


def move_to_numpy(obj):
    if type(obj) == list:
        npobj = []
        for object in obj:
            objnp = move_to_numpy(object)
            npobj.append(objnp)
        return npobj
    elif type(obj) == dict:
        npdict = dict()
        for key in obj:
            objnp = move_to_numpy(obj[key])
            npdict[key] = objnp 
        return npdict
    else:
        if type(obj) == torch.Tensor:
            return obj.cpu().detach().numpy()
        else:
            return obj


def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device 


def save_model(model, file):
    torch.save(model, file)


def pickle_save(data, file, to_numpy):
    with open(file, "wb") as h:
        if to_numpy is True:
            pickle.dump(move_to_numpy(data), h)
        else:
            pickle.dump(data, h)


def pickle_load(file):
    with open(file, "rb") as h:
        data = pickle.load(h)
    return data


def KL_bernoulli(q_params, prior_probs):
    q_prob = torch.sigmoid(q_params)
    q_logprob = F.logsigmoid(q_params)
    q_invlogprob = F.logsigmoid(-q_params)

    KL = torch.sum(q_prob*(q_logprob - torch.log(prior_probs))) + \
            torch.sum((1-q_prob)*(q_invlogprob - torch.log(1-prior_probs)))
    return KL


def hellinger_distance(x,y):
    return (1.0/np.sqrt(2.0))*np.sqrt(((np.sqrt(x) - np.sqrt(y))**2).sum(axis=-1))


# TODO: will want to take multiple model samples instead of using median beta,theta...
# TODO: compute stability metric on each sample
def get_basic_zprobs(data, beta_i, theta_i, beta_j, theta_j):
     #* compute vectors z = z(x)
    num_communities, _ = theta_i.shape
    num_particles, _ = data.shape

    zi = np.zeros((num_communities, num_particles))
    zj = np.zeros((num_communities, num_particles))

    #* compute particle usage probs...
    # TODO: vectorize so easy to take mulitple samples of metric...
    for kidx in range(num_communities):
        for lidx in range(num_particles):
            log_posterior_i = np.log(beta_i[kidx] + 1e-20) + np.sum(data[lidx,:]*np.log(theta_i[kidx,:] + 1e-20))
            log_posterior_j = np.log(beta_j[kidx] + 1e-20) + np.sum(data[lidx,:]*np.log(theta_j[kidx,:] + 1e-20))

            zi[kidx,lidx] = log_posterior_i
            zj[kidx,lidx] = log_posterior_j
    return zi, zj


def get_perturbation_zprobs(data, beta_i, theta_i, beta_j, theta_j):
    #* compute vectors z = z(x)
    num_communities, num_otus = theta_i.shape

    times = list(data.keys()) # times or groups
    num_times = len(times)
    subjs = list(data[times[0]].keys())
    nsubjs = len(subjs)
    num_particles = 0
    for t in times:
        for s in subjs:
            npart_local = data[t][s].shape[0]
            num_particles += npart_local

    print(f"{num_particles} total particles...")
    #* get number time points, number subjects, and total number of particles...
    zi = np.zeros((num_communities, num_particles)) #* size L x N (number of latents x number particles)
    zj = np.zeros((num_communities, num_particles))

    #* compute particle usage probs...
    # TODO: vectorize so easy to take mulitple samples of metric...
    for kidx in range(num_communities):
        lidx = 0
        for tind, t in enumerate(times):
            for sind, s in enumerate(subjs):
                for pind in range(data[t][s].shape[0]):
                    log_posterior_i = np.log(beta_i[kidx,tind,sind] + 1e-20) + np.sum(data[t][s][pind,:]*np.log(theta_i[kidx,:] + 1e-20))
                    log_posterior_j = np.log(beta_j[kidx,tind,sind] + 1e-20) + np.sum(data[t][s][pind,:]*np.log(theta_j[kidx,:] + 1e-20))

                    zi[kidx,lidx] = log_posterior_i
                    zj[kidx,lidx] = log_posterior_j
                    lidx += 1
    return zi, zj


def get_timeseries_zprobs(data, beta_i, theta_i, beta_j, theta_j):
    #* compute vectors z = z(x)
    num_communities, num_otus = theta_i.shape

    times = list(data.keys()) # times or groups
    num_times = len(times)
    num_particles = 0
    for t in times:
        npart_local = data[t].shape[0]
        num_particles += npart_local

    print(f"{num_particles} total particles...")
    #* get number time points, number subjects, and total number of particles...
    zi = np.zeros((num_communities, num_particles)) #* size L x N (number of latents x number particles)
    zj = np.zeros((num_communities, num_particles))

    #* compute particle usage probs...
    # TODO: vectorize so easy to take mulitple samples of metric...
    for kidx in range(num_communities):
        lidx = 0
        for tind, t in enumerate(times):
            for pind in range(data[t].shape[0]):
                log_posterior_i = np.log(beta_i[kidx,tind] + 1e-20) + np.sum(data[t][pind,:]*np.log(theta_i[kidx,:] + 1e-20))
                log_posterior_j = np.log(beta_j[kidx,tind] + 1e-20) + np.sum(data[t][pind,:]*np.log(theta_j[kidx,:] + 1e-20))
                zi[kidx,lidx] = log_posterior_i
                zj[kidx,lidx] = log_posterior_j
                lidx += 1
    return zi, zj


def compute_similarity_matrix(data, beta_i, theta_i, beta_j, theta_j):
    """compute similarity matrix between models i and j

    Args:
        data (_type_): use data to get map community assignments of each particle
        beta_i (_type_): _description_
        theta_i (_type_): _description_
        beta_j (_type_): _description_
        theta_j (_type_): _description_
    """
    if type(data) == dict:
        keys = list(data.keys())
        if type(data[keys[0]]) == dict:
            zi, zj = get_perturbation_zprobs(data, beta_i, theta_i, beta_j, theta_j)
        else:
            zi, zj = get_timeseries_zprobs(data, beta_i, theta_i, beta_j, theta_j)
    else:   
        zi, zj = get_basic_zprobs(data, beta_i, theta_i, beta_j, theta_j)

    #*normalize log probs
    zi = zi - logsumexp(zi,axis=0)
    zj = zj - logsumexp(zj,axis=0)

    #* compute correlation matrix, take spearman correlation over num_particles N
    corrmat = spearmanr(zi, zj, axis=1).correlation

    #* take an off diagonal block... 
    num_communities, _ = theta_i.shape
    rmat = corrmat[:num_communities,num_communities:] 
    return rmat


def compute_stability_score(rij, gamma_i, gamma_j):
    IKL_a = gamma_i
    IKL_b = gamma_j
    d_a = IKL_a.sum()
    d_b = IKL_b.sum() 
    r_a = np.amax(rij, axis=0) 
    r_b = np.amax(rij, axis=1)
    summand_b = ((r_a**2)*IKL_b)/(rij.sum(axis=0) + 1e-20)
    summand_a = ((r_b**2)*IKL_a)/(rij.sum(axis=1) + 1e-20)
    sij = (1.0/(d_a + d_b))*( summand_b.sum() + summand_a.sum() )
    return sij


# TODO: take samples
def get_model_pair_stability_score(data, model_i, model_j, n_samples=1000):
    pass
