import numpy as np 
import pandas as pd 
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from composition_stats import ilr, ilr_inv
from sklearn.metrics import auc


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


def get_summary_results(model, data, n_samples=1000):
    if model.use_sparse_weights is True:
        gamma_probs = np.concatenate([[1],model.beta_params.sparsity_params.q_probs.cpu().detach().clone().numpy()])
    else:
        gamma_probs = np.ones(model.num_assemblages)
    #* using 95th percentile for model selection
    gammasub = (gamma_probs>0.95)

    if model.num_perturbations > 0:
        pert_probs = model.beta_params.perturbation_indicators.q_probs.cpu().detach().clone().numpy()
        pert_prior = model.perturbation_prior_prob
        pertprobsub = pert_probs[gammasub,:]
        pert_bf = get_bayes_factors(pertprobsub, pert_prior)
    else:
        pert_bf = None

    loss, theta, beta, gamma, pi = model(data)
    if pi is not None:
        ngrps = pi.shape[0]
        pi_samples = np.zeros((n_samples, ngrps))
    
    loss_samples = np.zeros((n_samples,))
    ncomm, ntime, nsubj = beta.shape
    ncomm_keep = gammasub.sum()
    beta_samples = np.zeros((n_samples, ncomm_keep, ntime, nsubj))
    fixed_gamma = torch.zeros_like(gamma)
    fixed_gamma[gammasub] = 1
    for i in range(n_samples):
        loss, _, _, _, pi = model(data)
        loss_samples[i] = loss.cpu().detach().clone().numpy()
        if pi is not None:
            pi_samples[i,:] = pi.cpu().detach().clone().numpy()
        # *need latent beta
        x_latent = model.beta_params.x_latent
        beta = sparse_softmax(x_latent, fixed_gamma)
        beta = beta.cpu().detach().clone().numpy()
        betasub = beta[gammasub,:,:]
        # **renomalize
        betasub /= betasub.sum(axis=0, keepdims=True)
        beta_samples[i,:,:,:] = betasub
    beta_summary = np.mean(beta_samples, axis=0)
    
    theta = theta.cpu().detach().clone().numpy()
    thetasub = theta[gammasub,:]
    if pi is not None:
        pi_summary = np.mean(pi_samples, axis=0)
    else:
        pi_summary = None
    mean_loss = np.mean(loss_samples)
    return pert_bf, beta_summary, thetasub, pi_summary, mean_loss


def get_posterior_summary_data(model, data, taxonomy, times, subjects):
    pert_bf, beta_summary, theta_summary, pi_summary, mean_loss = get_summary_results(model, data)
    ncomm, ntime, nsubj = beta_summary.shape
    _, npert = pert_bf.shape
    assemblages = [f"A{i+1}" for i in range(ncomm)]

    betatimes = []
    betasubj = []
    betaval = []
    betaassem = []
 
    for i,t in enumerate(times):
        for j,s in enumerate(subjects):
            for k,a in enumerate(assemblages):
                betatimes.append(t)
                betasubj.append(s)
                betaassem.append(a)
                betaval.append(beta_summary[k,i,j])
    betadf = pd.DataFrame({'Time': betatimes, 'Subject': betasubj,
                      'Assemblage': betaassem, 'Value': betaval})
    betadf['log10Value'] = np.log10(betadf['Value'] + 1e-20)
    
    multiind = pd.MultiIndex.from_frame(taxonomy.reset_index())
    thetadf = pd.DataFrame(theta_summary.T, index=multiind, columns=assemblages)

    pertdf = pd.DataFrame(pert_bf, index=assemblages, columns=[f'P{i+1}' for i in range(npert)])
    return thetadf, betadf, pertdf


def get_sig_perturbation_diff_subset(betadf, pertdf, pidx, t_after, t_before, bf_threshold=10):
    b_before = betadf.loc[betadf['Time'] == t_before]
    b_after = betadf.loc[betadf['Time'] == t_after]
    
    b_before_multi = b_before.set_index(['Subject', 'Assemblage'])[['Value']]
    b_after_multi = b_after.set_index(['Subject', 'Assemblage'])[['Value']]
    
    bdiff = b_after_multi.subtract(b_before_multi).reset_index()
    
    sigpertinds = pertdf.loc[pertdf[f'P{pidx}']>=bf_threshold, f'P{pidx}'].index
    bdiffsub = bdiff.loc[bdiff['Assemblage'].isin(sigpertinds),:]
    meandiff = bdiffsub[['Assemblage', 'Value']].groupby('Assemblage').mean()
    bpertorder = meandiff.sort_values(by='Value').index
    return bdiffsub, bpertorder


def get_pert_otu_sub(thetadf, pertcomms, otu_threshold = 0.01):
    sigpertcomms = thetadf.reset_index().set_index('Otu')[pertcomms]
    otusub = sigpertcomms.loc[(sigpertcomms>0.01).any(axis=1),:].index
    return otusub


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


def estimate_process_variance(reads, num_otus, subjects, sample_times):
    #* get an average time step
    t_temp = np.array(sample_times)
    dt = np.mean(t_temp[1:] - t_temp[:-1])

    num_samp_times = len(sample_times)
    #* estimate variance for all subjects
    timevars = {}
    for s in subjects:
        tdata = np.zeros((num_otus, num_samp_times))
        for i, tm in enumerate(sample_times):
            counts = reads[tm][s]
            ra = counts.sum(axis=0)/counts.sum()
            tdata[:,i] = np.log(ra + 1e-20)
        tramean = np.mean(tdata, axis=1)
        tvar = np.var(tdata, axis=1)
        tvar = tvar[tramean>np.log(0.005)]
        tvarmed = np.median(tvar)
        timevars[s] = tvarmed

    #* take median over subjects
    medvar = np.median(np.array(list(timevars.values())))
    return medvar/dt


def ilr_transform_data(data):
    EPS = 1e-8 #* below which data is considered to be '0'

    relative_abundance = data/(data.sum(axis=1, keepdims=True))
    
    #* deal with zeros
    nparticles, notus = relative_abundance.shape
    delta = 1.0/(notus**2) # imputed value
    
    zdata = np.zeros((nparticles, notus))
    for lidx in range(nparticles):
        zeros = (relative_abundance[lidx,:] < EPS)
        nzeros = zeros.sum()
        zdata[lidx,:] = (1.0 - nzeros*delta)*relative_abundance[lidx,:]
        zdata[lidx,zeros] = delta

    tdata = ilr(zdata)
    return tdata


def inv_ilr_transform_data(data):
    return ilr_inv(data)


def flatten_data(data):
    reads = data['reads']
    assign = data['assignments']
    groups = list(reads.keys())

    flatreads = None
    subj_labels = None #[]
    cluster_labels = None
    
    for ig, grp in enumerate(groups):
        subjs = list(reads[grp].keys())
        for isx, sub in enumerate(subjs):
            particles = reads[grp][sub]
            nparticles, notus = particles.shape
            clabs = assign[grp][sub]
            if flatreads is None:
                flatreads = particles
                cluster_labels = clabs
            else:
                flatreads = np.vstack([flatreads, particles])
                cluster_labels = np.concatenate([cluster_labels, clabs])
            for _ in range(nparticles):
                if subj_labels is None:
                    subj_labels = np.array([ig, isx])
                else:
                    subj_labels = np.vstack([subj_labels, np.array([ig, isx])])
                # gslabel = f"{grp}{sub}"
                # subj_labels.append(gslabel)
        
    return flatreads, subj_labels, cluster_labels


def get_mcspace_cooccur_prob(model, data, otu_threshold, nsamples=100):
    # time_idx, subj_idx,
    #! return full tensor over all times and subjects (or return dict??)
    cooccur_prob = 0
    for i in range(nsamples):
        loss, theta, beta, gamma, _ = model(data)
        theta = theta.cpu().clone().detach().numpy()
        beta = beta.cpu().clone().detach().numpy()
        gamma = gamma.cpu().clone().detach().numpy()
        _, ntime, nsubj = beta.shape
        summand = gamma[:,None,None,None,None]*beta[:,:,:,None,None]*(theta[:,None,None,None,:] > otu_threshold)*(theta[:,None,None,:,None] > otu_threshold)
        prob_sample = summand.sum(axis=(0,1,2))/(nsubj*ntime)
        cooccur_prob += prob_sample
    cooccur_prob /= nsamples
    return cooccur_prob


def get_gt_assoc(theta, otu_threshold):
    K, notus = theta.shape
    gt_assoc = 0
    for kidx in range(K):
        gt_assoc += np.outer(theta[kidx,:] > otu_threshold, theta[kidx,:] > otu_threshold)
    for oidx in range(notus):
        # remove self-assoc
        gt_assoc[oidx,oidx] = 0 
    
    gt_assoc[gt_assoc>0.5] = 1 #* 1 or 0; don't need more than 1

    return gt_assoc


def calc_auc(gt_assoc, post_probs, nthres = 100):
    notus = gt_assoc.shape[0]
    # take upper triangular matrices
    gta = (gt_assoc[np.triu_indices(notus, k=1)] > 0.5)
    pp = post_probs[np.triu_indices(notus, k=1)]
    thresholds = np.linspace(-0.001, 1.001, nthres)

    true_pos = np.zeros((nthres,))
    false_pos = np.zeros((nthres,))
    true_neg = np.zeros((nthres,))
    false_neg = np.zeros((nthres,))
    
    for i,thres in enumerate(thresholds):
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
    return auc_val, true_pos, false_pos, true_neg, false_neg


def get_min_loss_path(runpath, seeds):
    losses = {}

    seeds = np.arange(10)
    for seed in seeds:
        respath = runpath / f"seed_{seed}"
        model = torch.load(respath / MODEL_FILE)
        data = pickle_load(respath / DATA_FILE)

        n_samples = 100
        loss_samples = np.zeros(n_samples)
        for i in range(n_samples):
            loss, _, _, _, _ = model(data)
            loss_samples[i] = loss.item()
        losses[seed] = np.mean(loss_samples)
        print(seed)
    
    best_seed = min(losses, key=losses.get)
    print(best_seed)
    respath = runpath / f"seed_{best_seed}"
    return respath


def apply_taxonomy_threshold(taxonomy, threshold=0.5):
    ranks = ['domain', 'phylum', 'class', 'order', 'family', 'genus']
    conf = ['dconf', 'pconf', 'cconf', 'oconf', 'fconf', 'gconf']
    
    taxcopy = taxonomy.reset_index()
    ntaxa = taxcopy.shape[0]
    for i in range(ntaxa):
        for r,c in zip(ranks, conf):
            if taxcopy.loc[i,c] < threshold:
                taxcopy.loc[i,r] = 'na'
    
    ptaxa = taxcopy.set_index("Otu")
    ptaxa2 = ptaxa[ranks]
    
    mapper = {x:x.capitalize() for x in list(ptaxa2.columns)}
    ptaxa3 = ptaxa2.rename(columns=mapper)
    return ptaxa3


def get_abundance_order(betadf):
    betadf_drop = betadf[['Assemblage', 'Value']]
    aveval = betadf_drop.groupby('Assemblage').mean()
    beta_order = aveval.sort_values(by='Value', ascending=False).index
    return beta_order
