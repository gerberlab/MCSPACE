import numpy as np 
import pandas as pd 
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_FILE = "model.pt"
RESULT_FILE = "results.pkl"
DATA_FILE = "data.pkl"


def BernoulliKL(q_logits, prior_p):
    q = torch.sigmoid(q_logits)
    log_q = F.logsigmoid(q_logits)
    log_one_minus_q = F.logsigmoid(-q_logits)

    KL = ((1-q)*(log_one_minus_q - torch.log(1.0 - prior_p))) \
        + (q*(log_q - torch.log(prior_p)))
    return KL


def BernoulliKLPower(q_logits, log_prior_p, pow):
    q = torch.sigmoid(q_logits)
    log_q = F.logsigmoid(q_logits)
    log_one_minus_q = F.logsigmoid(-q_logits)
    prior_p = torch.exp(pow*log_prior_p)

    KL = ((1-q)*(log_one_minus_q - torch.log(1.0 - prior_p))) \
        + (q*(log_q - pow*log_prior_p))
    return KL


def GaussianKL(mu, var, prior_mu, prior_var):
    KL = 0.5*( torch.log(prior_var) - torch.log(var) + (var/prior_var) \
              + (((prior_mu-mu)**2)/prior_var) - 1.0 )
    return KL


def sparse_softmax(x, gamma):
    a = torch.amax(x,dim=0)
    temp = gamma[:,None,None]*torch.exp(x - a)
    res = temp/temp.sum(dim=0)
    if (torch.isnan(res).any()) or (torch.isinf(res).any()):
        raise ValueError("nan or inf in sparse softmax") 
    return res
    

def inverse_softplus(x):
    return np.log(np.exp(x) - 1.0)


def pickle_save(file, data):
    with open(file, "wb") as h:
        pickle.dump(data, h)


def pickle_load(file):
    with open(file, "rb") as h:
        data = pickle.load(h)
    return data


def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def hellinger_distance(x,y):
    return (1.0/np.sqrt(2.0))*np.sqrt(((np.sqrt(x) - np.sqrt(y))**2).sum(axis=-1))


def get_cosine_dist(x,y):
    # x shape LxO, y shape LxO; return L
    # 1 - cossim
    # cossim = x.y / |x||y|
    xnorms = np.linalg.norm(x, axis=1)
    ynorms = np.linalg.norm(y, axis=1)
    xdoty = np.sum(x*y, axis=1)
    cossim = xdoty/(xnorms*ynorms)
    return 1.0 - cossim


def sample_assignments(otu_dist, comm_dist, counts):
    EPS = 1e-6
    num_communities, _ = otu_dist.shape
    num_particles, _ = counts.shape

    #* compute posterior assignment probabilities
    logprob = (counts[:,None,:]*np.log(otu_dist[None,:,:] + EPS)).sum(axis=-1) + np.log(comm_dist[None,:] + EPS)
    
    for kidx in range(num_communities):
        if comm_dist[kidx] < EPS: 
            logprob[:,kidx] = -np.inf

    #* sample from categorical
    g = np.random.gumbel(size=(num_particles, num_communities))
    z = np.argmax(g + logprob, axis=1)
    return z


def sample_reads(theta, assigns, read_depths):
    # assigns of size L=particles
    ncomm, notus = theta.shape
    npart = len(assigns)
    reads = np.zeros((npart, notus))
    for lidx in range(npart):
        dist = theta[assigns[lidx],:]
        dist = np.squeeze(np.asarray(dist).astype('float64'))
        dist = dist / np.sum(dist)
        rd = read_depths[lidx]
        rsamp = np.random.multinomial(rd, pvals=dist)
        reads[lidx,:] = rsamp
    return reads


def get_bayes_factors(post_prob, prior_prob):
    post_odds = post_prob/(1.0-post_prob)
    inv_prior_odds = (1.0-prior_prob)/prior_prob
    return post_odds*inv_prior_odds


def get_summary_stats(model, data, n_samples = 1000):
    # return sparse: pert bayes factors, beta, and theta
    gamma_probs = np.concatenate([[1],model.beta_params.sparsity_params.q_probs.cpu().detach().clone().numpy()])
    gammasub = (gamma_probs>0.5)

    if model.num_perturbations > 0:
        pert_probs = model.beta_params.perturbation_indicators.q_probs.cpu().detach().clone().numpy()
        pert_prior = model.perturbation_prior_prob
        pertprobsub = pert_probs[gammasub,:]
        pert_bf = get_bayes_factors(pertprobsub, pert_prior)
    else:
        pert_bf = None

    loss, theta, beta, gamma = model(data)
    ncomm, ntime, nsubj = beta.shape
    beta_samples = np.zeros((n_samples, ncomm, ntime, nsubj))
    for i in range(n_samples):
        loss, theta, beta, gamma = model(data)
        beta_samples[i,:] = beta.cpu().detach().clone().numpy()
    beta_mean = np.mean(beta_samples, axis=0)
    betameansub = beta_mean[gammasub,:,:]
    betameansub = betameansub/betameansub.sum(axis=0, keepdims=True)
    
    theta = theta.cpu().detach().clone().numpy()
    thetasub = theta[gammasub,:]
    return pert_bf, betameansub, thetasub


def down_sample_reads_percentage(reads, percentage, threshold=-1, replace=False):
    npart, notus = reads.shape 
    new_reads = np.zeros((npart, notus))
    
    for lidx in range(npart):
        rd = reads[lidx,:].sum()
        new_read_depth = int(percentage*rd)
        preads = down_sample_reads_particle(lidx, reads, new_read_depth, replace)
        new_reads[lidx,:len(preads)] = preads 
    
    # remove particles below threshold of reads...
    rdfull = new_reads.sum(axis=1)
    new_reads = new_reads[rdfull>threshold,:]
    return new_reads


def down_sample_reads_particle(lidx, reads, new_read_depth, replace=False):
    # for each particle, sample reads without replacement
    npart, notus = reads.shape 
    # resample reads from each particle with given read depth
    rd = int(np.sum(reads[lidx,:]))
    otu_counts = np.zeros((rd,), dtype=np.int64)
    k = 0 
    # ''flatten'' reads, with mulitple copies of each otu-index
    for oidx in range(notus):
        ncounts = int(reads[lidx,oidx])
        for _ in range(ncounts):
            otu_counts[k] = oidx 
            k=k+1 
    sampled = np.random.choice(otu_counts, new_read_depth, replace=replace)
    preads = np.bincount(sampled)  # reads for each otus
    return preads 
