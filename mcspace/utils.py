import numpy as np 
import pandas as pd 
import torch
import torch.nn.functional as F
import pickle 
from scipy.stats import spearmanr #, hypergeom
from scipy.special import softmax, logsumexp
from statsmodels.stats.multitest import multipletests
import scipy.stats as stats
from scipy.stats import hypergeom
from sklearn.metrics import auc


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


def sample_assignments(otu_dist, comm_dist, contam_weight, contam_comm, counts):
    EPS = 1e-6
    num_communities, _ = otu_dist.shape
    num_particles, _ = counts.shape

    #* compute posterior assignment probabilities
    otu_mixture_distrib = otu_dist*(1-contam_weight) + contam_weight*contam_comm
    logprob = (counts[:,None,:]*np.log(otu_mixture_distrib[None,:,:] + EPS)).sum(axis=-1) + np.log(comm_dist[None,:] + EPS)
    
    #! due to numerical handling.... need to finalize this throughout each part of model!!!
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


def get_cosine_dist(x,y):
    # x shape LxO, y shape LxO; return L
    # 1 - cossim
    # cossim = x.y / |x||y|
    xnorms = np.linalg.norm(x, axis=1)
    ynorms = np.linalg.norm(y, axis=1)
    xdoty = np.sum(x*y, axis=1)
    cossim = xdoty/(xnorms*ynorms)
    return 1.0 - cossim


def get_euclidean_dist(x,y):
    return np.sqrt(np.sum(np.square(x-y), axis=1))


def get_spearman_corr(x,y):
    npart = x.shape[0]
    spc = np.zeros(npart)
    for lidx in range(npart):
        spc[lidx] = spearmanr(x[lidx,:],y[lidx,:]).statistic
    return spc


#* model summary
def get_bayes_factors(indicator, prior_prob):
    post_odds = indicator/(1.0-indicator)
    inv_prior_odds = (1.0-prior_prob)/prior_prob
    return post_odds*inv_prior_odds


def get_perturbation_summary_results(model, data): # prior_gamma = None *could do this for 'default'
    params = model.community_distribution.get_params()
    gamma = np.concatenate([[1], params['sparsity_probs'].cpu().detach().clone().numpy()])

    beta = params['beta']
    using_comparator = False
    if len(beta.shape) == 3:
        n_comm, n_grps, n_subj = beta.shape
        if n_grps == 3:
            using_comparator = True
        n_time = 1
    else:
        n_comm, n_grps = beta.shape
        n_subj = 1
        n_time = (n_grps-1) # TODO: double check prior setting in model

    n_otus = model.num_otus

    # TODO: default value -- might want to give option to change this
    prior_gamma = 0.5/(n_comm - 1) 
    prior_perturbation = 0.5/(n_comm*n_time)

    bf_gamma = get_bayes_factors(gamma, prior_gamma)
    # keep communities with sparsity bayes factor > 100 (strong evidence)
    comm_sub = (bf_gamma > 100)

    pert_inds = params['perturbation_indicators'].cpu().detach().clone().numpy()
    bf_pert = get_bayes_factors(pert_inds, prior_perturbation)

    if using_comparator:
        comparator_pert_inds = params['comparator_perturbation_indicators'].cpu().detach().clone().numpy()
        bf_pert_comparator = get_bayes_factors(comparator_pert_inds, prior_perturbation)
        bf_pert_comparator = bf_pert_comparator[:,None]
    else:
        bf_pert_comparator = None

    if len(pert_inds.shape) == 1:
        pert_inds = pert_inds[:,None]
        bf_pert = bf_pert[:,None]

    # get posterior mean of communities
    n_samples = 1000
    theta_samples = np.zeros((n_samples, n_comm, n_otus))
    beta_samples = np.zeros((n_samples, n_comm, n_grps))
    beta_subj_samples = np.zeros((n_samples, n_comm, n_grps, n_subj))
    pert_effect_samples = np.zeros((n_samples, n_comm, n_time))
    pi_contam_samples = np.zeros((n_samples, n_grps))
    loss_samples = np.zeros(n_samples)
    if using_comparator:
        comparator_pert_effect_samples = np.zeros((n_samples, n_comm, n_time)) #! LEFT OFF HERE

    # take model samples
    for i in range(n_samples):
        loss, theta, beta, pi, _ = model(data)
        loss_samples[i] = loss.cpu().detach().clone().numpy()
        pi_contam_samples[i,:] = pi.cpu().detach().clone().numpy()
        params = model.community_distribution.get_params()
        beta_mean = params['beta_mean']
        pert_mag = params['perturbation_magnitude']
        theta_samples[i,:] = theta.cpu().detach().clone().numpy()
        beta_samples[i,:] = np.squeeze(beta_mean.cpu().detach().clone().numpy())
        beta = beta.cpu().detach().clone().numpy()
        if len(beta.shape) == 2:
            beta = beta[:,:,None]
        beta_subj_samples[i,:] = beta
        pert_mag = pert_mag.cpu().detach().clone().numpy()
        if len(pert_mag.shape) == 1:
            pert_mag = pert_mag[:,None]
        pert_effect_samples[i,:] = pert_mag
        if using_comparator:
            comp_pert_mag = params['comparator_perturbation_magnitude'].cpu().detach().clone().numpy()
            comparator_pert_effect_samples[i,:,0] = comp_pert_mag

    # take posterior mean
    theta_mean = np.mean(theta_samples, axis=0)
    beta_mean = np.mean(beta_samples, axis=0)
    beta_subj_mean = np.mean(beta_subj_samples, axis=0)
    pert_effect_mean = np.mean(pert_effect_samples, axis=0)
    pi_contam_mean = np.mean(pi_contam_samples, axis=0)
    loss_mean = np.mean(loss_samples)
    if using_comparator:
        comparator_pert_effect_mean = np.mean(comparator_pert_effect_samples, axis=0)
    else:
        comparator_pert_effect_mean = None

    # subset communities with strong evidence of presence
    theta_subset = theta_mean[comm_sub,:]
    beta_subset = beta_mean[comm_sub,:]
    beta_subj_subset = beta_subj_mean[comm_sub,:]
    pert_effect_subset = pert_effect_mean[comm_sub,:]
    bf_pert_subset = bf_pert[comm_sub,:]
    pert_inds_subset = pert_inds[comm_sub,:]
    bf_gamma_subset = bf_gamma[comm_sub]
    if using_comparator:
        bf_pert_comparator_subset = bf_pert_comparator[comm_sub,:]
        comparator_pert_effect_subset = comparator_pert_effect_mean[comm_sub,:]

    # order by perturbation effect size
    comm_order = np.argsort(((bf_pert_subset > 10)*(pert_effect_subset)).sum(axis=1))

    theta_ord = theta_subset[comm_order,:]
    beta_ord = beta_subset[comm_order,:]
    beta_subj_ord = beta_subj_subset[comm_order,:]
    pert_inds_ord = pert_inds_subset[comm_order,:]
    bf_pert_ord = bf_pert_subset[comm_order,:]
    pert_effect_ord = pert_effect_subset[comm_order,:]
    bf_gamma_ord = bf_gamma_subset[comm_order]
    if using_comparator:
        bf_pert_comparator_ord = bf_pert_comparator_subset[comm_order,:]
        comparator_pert_effect_ord = comparator_pert_effect_subset[comm_order,:]
    else:
        bf_pert_comparator_ord = None
        comparator_pert_effect_ord = None

    # collect results in dict
    results = {}
    results['theta_ord'] = theta_ord
    results['beta_ord'] = beta_ord
    results['beta_subj_ord'] = beta_subj_ord
    results['pert_inds_ord'] = pert_inds_ord
    results['bf_pert_ord'] = bf_pert_ord
    results['pert_effect_ord'] = pert_effect_ord
    results['bf_gamma_ord'] = bf_gamma_ord
    results['pi_contam_mean'] = pi_contam_mean
    results['loss_mean'] = loss_mean
    results['bf_pert_comparator_ord'] = bf_pert_comparator_ord
    results['comparator_pert_effect_ord'] = comparator_pert_effect_ord
    return results

# main differencs: subjects vs one subject; size of pert-inds; rest same
def get_time_series_summary_results(model, data):
    pass

def get_summary_results(model, data):
    # check model instance -- process accordingly...
    # TODO: this should be called at end of training and used to spit out final model results*
    #... can add model params, etc.. as well on top of this
    results = get_perturbation_summary_results(model, data)
    return results


#* enrichment analysis
def hypergeo_cdf(x,M,n,N):
    # TODO: comment on what each parameter means...
    prb = hypergeom.cdf(x, N, M, n)
    return prb

def compute_pvalues(df, dfcount):
    dfcount = dfcount.loc[df.index,:]
    ntaxa, nmods = df.shape
    pvalues = np.zeros((ntaxa,nmods))
    for taxid in range(ntaxa):
        for modid in range(nmods):
            k = df.iloc[taxid, modid]
            N = dfcount.values.sum()
            M = dfcount.iloc[taxid,:].sum()
            n = df.iloc[:,modid].sum()
            pv = 1-hypergeo_cdf(k-1,M,n,N)
            pvalues[taxid,modid] = pv 
    pvdf = pd.DataFrame(data=pvalues, index=df.index, columns=df.columns)
    return pvdf


def get_enrichment_results(theta, taxonomy, threshold = 0.005):
    # only subset of ranks that make sense for enrichment
    ranks = ["Phylum", "Class", "Order", "Family"]
    taxa_sub = taxonomy.loc[:,ranks]
    multiind = pd.MultiIndex.from_frame(taxa_sub)
    dfcomm = pd.DataFrame(data=theta.T, index=multiind)
    dfcount = pd.DataFrame(data=np.ones(taxa_sub.shape[0]), index=pd.MultiIndex.from_frame(taxa_sub))
    dfbin = dfcomm > threshold
    
    # # remove communities with fewer than 4 taxa
    # dfbin = dfbin.loc[:,dfbin.sum(axis=0)>=4]

    # compute enrichment and counts for each taxonomy level
    enrichment_pvals = {}
    enrichment_counts = {}
    raw_pvals = {}
    
    for level in ["Phylum", "Class", "Order", "Family"]:
        dflevel = dfbin.groupby(level=level).sum()
        dfcounttaxa = dfcount.groupby(level=level).sum()

        # remove with no counts at all
        subset = (dflevel.sum(axis=1) > 0)
        dflevel = dflevel.loc[subset,:]
        dfcounttaxa = dfcounttaxa.loc[subset,:]


        # dfcount = dfcountall
        pvdf = compute_pvalues(dflevel, dfcounttaxa)

        #! fix issues
        if 'na' in pvdf.index:
            pvdf.loc['na',:] = 1
        # pvdf[pvdf==0] = 1

# TODO: *** fix multiple test correction (include 0's and na's?) -- not really testing those...
        # # _,pvalue_c,_,_=multipletests(pvdf.values.reshape(-1),alpha=0.05,method='fdr_bh')
        # pvdf_corrected = pvdf # pd.DataFrame(data=pvalue_c.reshape(pvdf.shape), index=pvdf.index, columns=pvdf.columns)

        _,pvalue_c,_,_=multipletests(pvdf.values.reshape(-1),alpha=0.05,method='fdr_bh')
        pvdf_corrected = pd.DataFrame(data=pvalue_c.reshape(pvdf.shape), index=pvdf.index, columns=pvdf.columns)


        enrichment_counts[level] = dflevel
        raw_pvals[level] = pvdf

        # corrected 
        to_keep = []
        for idx in pvdf_corrected.index:
            if (pvdf_corrected.loc[idx,:]<0.05).any():
                to_keep.append(idx)

        pvkeep = pvdf_corrected 
        if pvkeep.shape[0] > 0:
            enrichment_pvals[level] = pvkeep

    return raw_pvals, enrichment_pvals, enrichment_counts


# *** FOR PAIRWISE COMPARATORS ==============================================
def get_sig_assoc_fisher(data_df, threshold=0.005): #, min_appearance=0.1):
    sig=0.05
    databin = data_df > threshold # LxO df of T/F values
    # data=databin.loc[:,((databin>0).sum(axis=0)>(databin.shape[0]*min_appearance))] 
    data = databin.T 

    pvalue_a=[]
    or_a=[]
    for i in np.arange(1,data.shape[1]):
        for i2 in np.arange(i):
            table=pd.crosstab(data.T.iloc[i] > 0, data.T.iloc[i2] > 0)
            # print(table)
            if table.shape == (2,2):
                oddsratio,p_value=stats.fisher_exact(table, alternative = 'greater')
                pvalue_a.append(p_value)
                or_a.append(np.log2(oddsratio))
            else:
                pvalue_a.append(np.nan)
                or_a.append(0)

    #multiple testing correction
    #! deal with nans
    pvalue_a = np.array(pvalue_a)
    nonnaninds = ~np.isnan(pvalue_a)
    pvalue_a_clean = pvalue_a[nonnaninds]
    _,pvalue_c_clean,_,_=multipletests(pvalue_a_clean,alpha=sig,method='fdr_bh')
    pvalue_c = np.ones(len(pvalue_a))
    pvalue_c[nonnaninds] = pvalue_c_clean
    # print(str(np.sum([i<sig for i in pvalue_c])) + ' significant associations detected, p<'+str(sig)+', FDR corrected')

    #reshape into matrix format
    mat_pv=np.zeros(shape=(data.shape[1],data.shape[1]))
    learned_assoc=np.zeros(shape=(data.shape[1],data.shape[1]))
    mat_or=np.zeros(shape=(data.shape[1],data.shape[1]))
    mat_sig= pd.DataFrame(index=data.T.index, columns=data.T.index)
    cnt=0
    for i in np.arange(1,data.shape[1]):
        for i2 in np.arange(i): 
            # mat_pv[i,i2]=np.log10(pvalue_c[cnt])
            mat_pv[i,i2]=pvalue_c[cnt]
            mat_or[i,i2]=or_a[cnt]
            if pvalue_c[cnt] < sig:
                mat_sig.iloc[i,i2] = str('x')
                if (mat_or[i,i2] > 0):
                    learned_assoc[i,i2] = 1
            else:
                mat_sig.iloc[i,i2] = str('')
            cnt+=1
    index_p=data.T.index
    to_plot=pd.DataFrame(mat_or+mat_or.T,index=index_p, columns=index_p)
    mat_sig = mat_sig.T.replace(np.NaN,'') + mat_sig.replace(np.NaN,'')
    learned_assoc = learned_assoc + learned_assoc.T

    mat_pv[np.isnan(mat_pv)] = 1.0
    mat_pv = mat_pv + mat_pv.T
    for i in range(data.shape[1]):
        mat_pv[i,i] = 1.0
    return learned_assoc, to_plot, mat_sig, mat_pv


def calc_auc(gt_assoc, post_probs, nthres = 100):
    notus = gt_assoc.shape[0]
    # take upper triangular matrices
    gta = (gt_assoc[np.triu_indices(notus, k=1)] > 0.5)
    pp = post_probs[np.triu_indices(notus, k=1)]

    # thresholds = np.linspace(0.001, 0.999, nthres)
    # TODO: double check, issue with previous values is not getting the tn values at prob=0
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


def get_mcspace_posterior_cooccur_prob(model, data, otu_threshold, nsamples=1000):
#     model.train() # for sampling
#     loss, theta, beta, pi_garb = model(data)
#     num_communities, notus = theta.shape
    
    otu_cooocur_probs = 0
    for i in range(nsamples):
        loss, theta, beta, pi_garb, gamma = model(data)
        theta = theta.cpu().clone().detach().numpy()
        
        # TODO: check | gamma = (beta > 1e-6).cpu().clone().detach().numpy() # value of gamma sample... (should be able to get from model)
        
        # zl = np.random.choice(num_communities, p=beta)
        # otu_cooocur_probs += np.outer(theta[zl,:],theta[zl,:])
        smat = 0
        K = theta.shape[0]
        for kidx in range(K):
            smat += gamma[kidx]*(np.outer(theta[kidx,:] > otu_threshold, theta[kidx,:] > otu_threshold))
        otu_cooocur_probs += (smat>0.5) 
    probs = otu_cooocur_probs/nsamples
    return probs


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
