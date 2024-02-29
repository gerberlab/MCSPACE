import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax
# from mcspace.utils import KL_bernoulli # TODO: do we use this?; also add a KL_gaussian??? -- double check


#* sparsity scaling utilities
# ===========================================================================================
# TODO: come up with final method for tuning and simplify set of functions...
def count_dict_to_data_subjects(count_dict):
    full_counts = []

    for t in count_dict.keys():
        subjs = count_dict[t].keys()
        for s in subjs:
            full_counts.append(count_dict[t][s]) 
    combined_counts = torch.cat(full_counts, dim=0)
    return combined_counts

def count_dict_to_data_timepoints(count_dict):
    full_counts = []

    for t in count_dict.keys():
        full_counts.append(count_dict[t]) 
    combined_counts = torch.cat(full_counts, dim=0)
    return combined_counts

def count_dict_to_data(count_dict):
    key1 = list(count_dict.keys())[0]
    obj = count_dict[key1]

    if type(obj) == dict:
        return count_dict_to_data_subjects(count_dict)
    else:
        return count_dict_to_data_timepoints(count_dict)

def get_log_p_from_data(input):
    counts = input['count_data']
    if type(counts) == dict:
        count_data = count_dict_to_data(counts) #* basically concat all particles -> final L* x O
    else:
        count_data = counts
    bulk = count_data.sum(dim=0)/count_data.sum() 
    ll = torch.sum(count_data * torch.log(bulk + 1e-8))
    return ll





#* community distribution variables
# ===========================================================================================
class AnnealedParameter(nn.Module):
    def __init__(self, start_temp, end_temp):
        super().__init__()
        self.start_temp = start_temp 
        self.end_temp = end_temp 
        self.concrete_temperature = None 

    def set_temp(self, epoch, num_epochs):
        percent = epoch/num_epochs
        if percent < 0.1:
            self.concrete_temperature = self.start_temp
            return
        if (percent >= 0.1) & (percent <= 0.9):
            interp = (percent-0.1)/0.8
            self.concrete_temperature = (1.0-interp)*self.start_temp + interp*self.end_temp
            return
        self.concrete_temperature = self.end_temp


class SparsityIndicator(AnnealedParameter):
    def __init__(self, num_communities, scale_multiplier, start_temp, end_temp, device):
        super().__init__(start_temp=start_temp, end_temp=end_temp) 

        self.num_communities = num_communities-1
        self.device = device 

        self.gamma = None 
        self.p_gamma = torch.tensor(0.5/num_communities).to(self.device)
        self.log_p_gamma = torch.log(self.p_gamma).to(self.device)
        self.p_scale = None 

        self.multiplier_lambda = scale_multiplier
        print("multiplier lambda = ", self.multiplier_lambda)

        q_gamma_params = torch.zeros((self.num_communities,), requires_grad=True, device=device)
        torch.nn.init.uniform_(q_gamma_params)
        self.q_gamma_params = torch.nn.Parameter(q_gamma_params)
        # #! TESTING - LARGER INIT...
        # self.q_gamma_params = torch.nn.Parameter(torch.normal(2,1, size=(self.num_communities,), requires_grad=True, device=device))

    def set_KL_scale_from_data(self, data=None):
        if data is not None:
            logp = get_log_p_from_data(data)
            self.p_scale = -logp
        else:
            self.p_scale = 1
        print("set p scale = ", self.p_scale)

    def sample_gamma(self):
        p_log = F.logsigmoid(torch.stack((self.q_gamma_params, -self.q_gamma_params)))
        gamma_rest = gumbel_softmax(p_log, hard=True, dim=0, tau=self.concrete_temperature)[0]
        self.gamma = torch.cat((torch.tensor([1.0]).to(self.device), gamma_rest))
        if torch.isnan(self.gamma).any():
            raise ValueError("nan in gamma")

    def forward(self):
        self.sample_gamma()
        q = torch.sigmoid(self.q_gamma_params)
        self.q_probs = q
# TODO: double check KL calc and scale...
        KL_gamma = torch.sum(q*(F.logsigmoid(self.q_gamma_params) - self.log_p_gamma)) \
            + torch.sum((1-q)*(F.logsigmoid(-self.q_gamma_params) - torch.log(1.0-self.p_gamma)))
        KL_gamma = self.multiplier_lambda*self.p_scale*KL_gamma
        return self.gamma, KL_gamma


class GroupMeanDistribution(nn.Module):
    def __init__(self, num_communities, num_otus, device):
        super().__init__()
        self.num_communities = num_communities 
        self.num_otus = num_otus
        self.device = device 
        self.hidden_dim = 10 
        self.q_encode = nn.Sequential(
            nn.Linear(num_otus, self.hidden_dim),
            nn.Softplus(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Softplus()
        )
        self.q_mu_params = nn.Linear(self.hidden_dim, self.num_communities)
        self.q_var_params = nn.Linear(self.hidden_dim, self.num_communities)
        self.eps = None 

    def sample_eps(self):
        self.eps = torch.normal(0, 1, size=(self.num_communities,), device=self.device, requires_grad=False)

    def forward(self, input):
        self.sample_eps()
        # input is data from a single group, but over all subjects 
        # loop over subjects
        enc = 0
        n_subj = 0
        for subj in input.keys():
            enc += self.q_encode(input[subj]).mean(dim=0) # mean over particles L
            n_subj += 1
        enc /= n_subj

        mu = self.q_mu_params(enc)
        var = torch.exp(self.q_var_params(enc))
        x = mu + torch.sqrt(var)*self.eps 
        KL = 0.5*torch.sum(var + (mu**2) - 1.0 - torch.log(var))
        return x, KL 


class PerturbationMagnitude(nn.Module):
    def __init__(self, num_communities, num_otus, device):
        super().__init__()
        self.num_communities = num_communities 
        self.num_otus = num_otus
        self.hidden_dim = 10 
        self.device = device
        # inflated variance to give diffuse prior
        self.prior_var = 100
        self.q_encode = nn.Sequential(
            nn.Linear(num_otus, self.hidden_dim),
            nn.Softplus(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Softplus()
        )
        self.q_mu_params = nn.Linear(self.hidden_dim, self.num_communities)
        self.q_var_params = nn.Linear(self.hidden_dim, self.num_communities)
        self.eps = None

    def sample_eps(self):
        self.eps = torch.normal(0,1,size=(self.num_communities,),device=self.device,requires_grad=False) 

    def forward(self, pre_peturb_data, post_perturb_data):
        self.sample_eps()
        # pre_peturb_data = input['pre_perturb']
        # post_perturb_data = input['post_perturb']

        nsubj = 0
        enc_pre = 0
        for s in pre_peturb_data.keys():
            enc_pre += self.q_encode(pre_peturb_data[s]).mean(dim=0)
            nsubj += 1
        enc_pre /= nsubj

        nsubj = 0
        enc_post = 0
        for s in post_perturb_data.keys():
            enc_post += self.q_encode(post_perturb_data[s]).mean(dim=0)
            nsubj += 1
        enc_post /= nsubj
        enc = 0.5*(enc_pre + enc_post)

        mu = self.q_mu_params(enc)
        var = torch.exp(self.q_var_params(enc))
        x = mu + torch.sqrt(var)*self.eps 
        KL = 0.5*torch.sum(var/self.prior_var + ((mu**2)/self.prior_var) - 1.0 - torch.log(var/self.prior_var))
        return x, KL
    

class PerturbationIndicators(AnnealedParameter):
    def __init__(self, shape, prior_prob, start_temp, end_temp, device):
        super().__init__(start_temp=start_temp, end_temp=end_temp) 
        self.shape = shape
        self.device = device 
        self.gamma = None 
        self.p_gamma = torch.tensor(prior_prob).to(dtype=torch.float, device=self.device)
        self.log_p_gamma = torch.log(self.p_gamma)
        q_gamma_params = torch.zeros(shape, requires_grad=True, device=device)
        torch.nn.init.uniform_(q_gamma_params)
        self.q_gamma_params = torch.nn.Parameter(q_gamma_params)

    def sample_gamma(self):
        p_log = F.logsigmoid(torch.stack((self.q_gamma_params, -self.q_gamma_params)))
        self.gamma = gumbel_softmax(p_log, hard=True, dim=0, tau=self.concrete_temperature)[0]
        if torch.isnan(self.gamma).any():
            raise ValueError("nan in gamma")

    def forward(self):
        self.sample_gamma()
        q = torch.sigmoid(self.q_gamma_params)
        self.q_probs = q
        KL_gamma = torch.sum(q*(F.logsigmoid(self.q_gamma_params) - self.log_p_gamma)) + torch.sum((1-q)*(F.logsigmoid(-self.q_gamma_params) - torch.log(1.0-self.p_gamma)))
        return self.gamma, KL_gamma 


class GroupVariance(nn.Module):
    def __init__(self, num_otus, num_subjects, prior_mean, prior_variance, device):
        super().__init__()
        self.num_otus = num_otus
        self.num_subjects = num_subjects
        self.prior_mean = prior_mean
        self.prior_var = prior_variance
        self.device = device
        self.hidden_dim = 10 
        # initial encoder
        # list S: [LxO -> LxH]
        self.q_encode = nn.Sequential(
            nn.Linear(num_otus, self.hidden_dim),
            nn.Softplus(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Softplus()
        )
        # mix subjects
        # take mean over L first
        # S: {LxH} -> S: {1xH}_<L> -> 1 x S.H 
        # [S.H -> S.H]
        self.q_mix = nn.Sequential(
            nn.Linear(self.num_subjects*self.hidden_dim, self.num_subjects*self.hidden_dim),
            nn.Softplus(),
            nn.Linear(self.num_subjects*self.hidden_dim, self.num_subjects*self.hidden_dim),
            nn.Softplus()
        )
        self.q_mu_params = nn.Linear(self.num_subjects*self.hidden_dim, 1)
        self.q_var_params = nn.Linear(self.num_subjects*self.hidden_dim, 1)
        self.eps = None 

    def sample_eps(self):
        self.eps = torch.normal(0, 1, size=(1,), device=self.device, requires_grad=False)

    def forward(self, input):
        # TODO: understand amortized better, may not need data input here?; only learning a scalar parameter...
        self.sample_eps() 
        # input is for specific group, dict over subjects
        enc = torch.zeros(self.num_subjects, self.hidden_dim).to(self.device)
        s_ind = 0
        for s in input.keys():
            data = input[s]
            enc[s_ind,:] = self.q_encode(data).mean(dim=0)
            s_ind += 1
        
        # reshape to mix subject components 
        enc = torch.reshape(enc, (self.num_subjects*self.hidden_dim,))
        # pass through next encoder with subjects connected
        enc = self.q_mix(enc)
        # evaluate posterior mean and variance params
        mu = self.q_mu_params(enc)
        var = torch.exp(self.q_var_params(enc))
        x = mu + torch.sqrt(var)*self.eps 
        x = F.softplus(x)
        KL = 0.5*torch.sum(var/self.prior_var + ((mu-self.prior_mean)**2)/self.prior_var - 1.0 - torch.log(var/self.prior_var))
        return x, KL


class SubjectDistribution(nn.Module):
    def __init__(self, num_communities, num_subjects, num_otus, device):
        super().__init__()
        self.num_communities = num_communities
        self.num_subjects = num_subjects
        self.num_otus = num_otus
        self.device = device
        self.hidden_dim = 10 
        self.q_encode = nn.Sequential(
            nn.Linear(num_otus, self.hidden_dim),
            nn.Softplus(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Softplus()
        )
        self.q_mu_params = nn.Linear(self.hidden_dim, self.num_communities)
        self.q_var_params = nn.Linear(self.hidden_dim, self.num_communities)
        self.eps = None

    def sample_eps(self):
        self.eps = torch.normal(0, 1, size=(self.num_communities,self.num_subjects), device=self.device, requires_grad=False)

    def compute_KL(self, x, mu, var, prior_mu, prior_var):
        logq = torch.distributions.Normal(loc=mu, scale=torch.sqrt(var)).log_prob(x).sum() # sum over K and S?
        logp = torch.distributions.Normal(loc=prior_mu[:,None], scale=torch.sqrt(prior_var)).log_prob(x).sum() # sum over K and S?
        return ((logq - logp))

    def forward(self, input, prior_mu, prior_var):
        self.sample_eps() 
        # input is for specific group, dict over subjects
        enc = torch.zeros(self.num_subjects, self.hidden_dim).to(self.device)
        s_ind = 0
        for s in input.keys():
            data = input[s]
            enc[s_ind,:] = self.q_encode(data).mean(dim=0) # mean over particles in given subject
            s_ind += 1
        # evaluate posterior mean and variance params; reshape to NxSxK; permute to KxNxS
        mu = self.q_mu_params(enc).T
        var = torch.exp(self.q_var_params(enc)).T
        x = mu + torch.sqrt(var)*self.eps 
        KL_x = self.compute_KL(x, mu, var, prior_mu, prior_var)
        return x, KL_x


class InitialDistribution(nn.Module):
    def __init__(self, num_communities, num_otus, device):
        super().__init__()
        self.num_communities = num_communities 
        self.num_otus = num_otus
        self.device = device 
        self.hidden_dim = 10 
        self.q_encode = nn.Sequential(
            nn.Linear(num_otus, self.hidden_dim),
            nn.Softplus(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Softplus()
        )
        self.q_mu_params = nn.Linear(self.hidden_dim, self.num_communities)
        self.q_var_params = nn.Linear(self.hidden_dim, self.num_communities)
        self.eps = None 

    def sample_eps(self):
        self.eps = torch.normal(0, 1, size=(self.num_communities,), device=self.device, requires_grad=False)

    def forward(self, input):
        self.sample_eps()
        # input is data from initial time point
        enc = self.q_encode(input).mean(dim=0) # mean over particles L
        mu = self.q_mu_params(enc)
        var = torch.exp(self.q_var_params(enc))
        x = mu + torch.sqrt(var)*self.eps 
        KL = 0.5*torch.sum(var + (mu**2) - 1.0 - torch.log(var))
        return x, KL 
    

# TODO: see if we can use same class as basic perturbation model
class TimeSeriesPerturbationMagnitude(nn.Module):
    def __init__(self, num_communities, num_time, num_otus, device):
        super().__init__()
        self.num_communities = num_communities
        self.num_time = num_time
        self.num_otus = num_otus
        self.hidden_dim = 10 
        self.device = device
        # inflated variance to give diffuse prior
        self.prior_var = 100
        self.q_encode = nn.Sequential(
            nn.Linear(num_otus, self.hidden_dim),
            nn.Softplus(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Softplus()
        )
        self.q_mu_params = nn.Linear(self.hidden_dim, self.num_communities*self.num_time)
        self.q_var_params = nn.Linear(self.hidden_dim, self.num_communities*self.num_time)
        self.eps = None

    def sample_eps(self):
        self.eps = torch.normal(0,1,size=(self.num_communities,self.num_time),device=self.device,requires_grad=False) 

    def forward(self, input):
        self.sample_eps()
        data = input['full_normed_data']
        enc = self.q_encode(data).mean(dim=0)
        mu = self.q_mu_params(enc).view(self.num_communities,self.num_time)
        var = torch.exp(self.q_var_params(enc)).view(self.num_communities,self.num_time)
        x = mu + torch.sqrt(var)*self.eps 
        KL = 0.5*torch.sum(var/self.prior_var + ((mu**2)/self.prior_var) - 1.0 - torch.log(var/self.prior_var))
        return x, KL
    



#* community distribution classes
# ===========================================================================================
class BasicCommunityDistribution(nn.Module):
    def __init__(self, num_communities, num_otus, device, sparse_communities, scale_multiplier=1):
        super().__init__()
        self.num_communities = num_communities
        self.num_otus = num_otus
        self.device = device
        self.sparse_communities = sparse_communities
        self.sparsity_gamma = SparsityIndicator(num_communities, scale_multiplier, start_temp=0.5, end_temp=0.01, device=device)

        self.hidden_dim = 50
        self.q_encode = nn.Sequential(
            nn.Linear(num_otus, self.hidden_dim),
            nn.Softplus(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Softplus()
        )

        self.q_topic_mu_params = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.q_topic_var_params = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.z_params = nn.Linear(self.hidden_dim, self.num_communities)
        self.eps = None 

    def set_temps(self, epoch, num_epochs):
        self.sparsity_gamma.set_temp(epoch, num_epochs)

    def get_params(self):
        return {
            'beta': self.beta,
            'KL_comm': self.KL_comm,
            'KL_gamma': self.KL_gamma,
            'sparsity_probs': self.sparsity_probs
        }
    
    def sample_eps(self):
        self.eps = torch.normal(0, 1, size=(self.hidden_dim,), device=self.device, requires_grad=False)

    def compute_comm_z(self, data, sample):
        enc = self.q_encode(data)
        enc = enc.mean(dim=0)
        mu = self.q_topic_mu_params(enc)
        var = torch.exp(self.q_topic_var_params(enc))
        z = self.z_params(mu + torch.sqrt(var)*sample) 
        return mu, var, z
    
    def forward(self, inputs):
        normed_data = inputs['normed_data']
        self.sample_eps()
        mu, var, z_topicdist = self.compute_comm_z(normed_data, self.eps)

        if self.sparse_communities is True:
            gamma, KL_gamma = self.sparsity_gamma()
        else:
            gamma = 1
            KL_gamma = 0

        te = torch.exp(torch.squeeze(z_topicdist))*gamma
        topic_distrib = te/torch.sum(te)

        KL_comm = 0.5*torch.sum(var + mu**2 - 1.0 - torch.log(var))
        KL = KL_comm + KL_gamma

        #* get parameters to save
        self.beta = topic_distrib
        self.KL_comm = KL_comm
        self.KL_gamma = KL_gamma
        self.sparsity_probs = self.sparsity_gamma.q_probs
        return topic_distrib, KL, gamma


class PerturbationCommunityDistribution(nn.Module):
    def __init__(self, num_communities, num_otus, num_groups, num_subjects, subject_variance, perturbation_prior_prob, device, sparse_communities, scale_multiplier=1):
        super().__init__()
        self.num_communities = num_communities
        self.num_otus = num_otus
        self.device = device
        self.sparse_communities = sparse_communities
        self.sparsity_gamma = SparsityIndicator(num_communities, scale_multiplier, start_temp=0.5, end_temp=0.01, device=device)

        self.num_subjects = num_subjects
        self.num_groups = num_groups
        if (self.num_groups != 2) and (self.num_groups != 3):
            # TODO: ???
            #! NOTE: could include more than 3 too...
            raise ValueError("Number of total groups should be 2 or 3") 
        
        self.perturbation_prior_prob = perturbation_prior_prob

        #* group means
        self.eta_pre_perturb = GroupMeanDistribution(num_communities, num_otus, device)
        
        #* perturbation parameters
        self.delta = PerturbationMagnitude(num_communities, num_otus, device)
        self.perturb_indicators = PerturbationIndicators(shape=(num_communities,), prior_prob=self.perturbation_prior_prob, start_temp=0.5, end_temp=0.01, device=device)

        #* learned variances for each group
        self.pre_perturb_variance = GroupVariance(num_otus, num_subjects, prior_mean=subject_variance['pre_perturb'], prior_variance=100, device=device)
        self.post_perturb_variance = GroupVariance(num_otus, num_subjects, prior_mean=subject_variance['post_perturb'], prior_variance=100, device=device)

        #* subject community distributions
        self.pre_perturb_distrib = SubjectDistribution(num_communities, num_subjects, num_otus, device)
        self.post_perturb_distrib = SubjectDistribution(num_communities, num_subjects, num_otus, device)

        #! if including comparator group
        if self.num_groups == 3: 
            self.delta_comparator = PerturbationMagnitude(num_communities, num_otus, device)
            self.perturb_indicators_comparator = PerturbationIndicators(shape=(num_communities,), prior_prob=self.perturbation_prior_prob, start_temp=0.5, end_temp=0.01, device=device)
            # self.eta_comparator = GroupMeanDistribution(num_communities, num_otus, device)
            self.comparator_variance = GroupVariance(num_otus, num_subjects, prior_mean=subject_variance['comparator'], prior_variance=100, device=device)
            self.comparator_distrib = SubjectDistribution(num_communities, num_subjects, num_otus, device)

    def set_temps(self, epoch, num_epochs):
        self.sparsity_gamma.set_temp(epoch, num_epochs)
        self.perturb_indicators.set_temp(epoch, num_epochs)
        if self.num_groups == 3:
            self.perturb_indicators_comparator.set_temp(epoch, num_epochs)

    def get_params(self):
        return {
                'beta_mean': self.beta_mean,
                'beta': self.beta,
                'perturbation_magnitude': self.perturbation_magnitude,
                'perturbation_indicators': self.perturbation_indicators,
                'comparator_perturbation_magnitude': self.comparator_perturbation_magnitude,
                'comparator_perturbation_indicators': self.comparator_perturbation_indicators,
                'sparsity_probs': self.sparsity_probs,
                'mean_pre_perturbation_distribution': self.mean_pre_perturbation_distribution,
                'mean_comparator_distribution': self.mean_comparator_distribution,
                'mean_post_perturbation_distribution': self.mean_post_perturbation_distribution,
                'var_pre_perturbation': self.var_pre_perturbation,
                'var_comparator': self.var_comparator,
                'var_post_perturbation': self.var_post_perturbation,
                'KL_pre_perturb': self.KL_pre_perturb,
                'KL_delta_comparator': self.KL_delta_comparator,
                'KL_c_comparator': self.KL_c_comparator,
                'KL_delta': self.KL_delta,
                'KL_c': self.KL_c,
                'KL_var_pre_perturb': self.KL_var_pre_perturb,
                'KL_var_comparator': self.KL_var_comparator,
                'KL_var_post_perturb': self.KL_var_post_perturb,
                'KL_pre_perturb_subj': self.KL_pre_perturb_subj,
                'KL_comparator_subj': self.KL_comparator_subj,
                'KL_post_perturb_subj': self.KL_post_perturb_subj,
                'KL_gamma': self.KL_gamma
                }

    def sparse_softmax(self, x, gamma):
        a = torch.amax(x,dim=0)
        temp = gamma[:,None,None]*torch.exp(x - a)
        res = temp/temp.sum(dim=0)
        if (torch.isnan(res).any()) or (torch.isinf(res).any()):
            raise ValueError("nan or inf in sparse softmax") 
        return res
    
    def forward(self, input):
        #* input is normed data; no counts ; what about 'full data'; over all particles, for all samples 
        normed_data = input['normed_data']
        full_data = input['full_normed_data'] # should be Lfull x O array
 
        # sample pre-perturbation group means
        x_pre_perturb, KL_pre_perturb = self.eta_pre_perturb(normed_data['pre_perturb'])

        # sample perturbation magnitude and indicators
        delta, KL_delta = self.delta(normed_data['pre_perturb'],normed_data['post_perturb'])
        c_indicators, KL_c = self.perturb_indicators()

        # compute post-perturbation distribution
        x_post_perturb = x_pre_perturb + delta*c_indicators

        # sample subject variance for each group
        var_pre_perturb, KL_var_pre_perturb = self.pre_perturb_variance(normed_data['pre_perturb'])
        var_post_perturb, KL_var_post_perturb = self.post_perturb_variance(normed_data['post_perturb'])

        # sample subject distributions for each group
        x_pre_perturb_subj, KL_pre_perturb_subj = self.pre_perturb_distrib(normed_data['pre_perturb'], x_pre_perturb, var_pre_perturb)
        x_post_perturb_subj, KL_post_perturb_subj = self.post_perturb_distrib(normed_data['post_perturb'], x_post_perturb, var_post_perturb)

        # sample comparator group if using
        if self.num_groups == 3:
            # x_comparator, KL_comparator = self.eta_comparator(normed_data['comparator'])
            delta_comparator, KL_delta_comparator = self.delta_comparator(normed_data['pre_perturb'],normed_data['comparator'])
            c_indicators_comparator, KL_c_comparator = self.perturb_indicators_comparator()
            x_comparator = x_pre_perturb + delta_comparator*c_indicators_comparator
            var_comparator, KL_var_comparator = self.comparator_variance(normed_data['comparator'])
            x_comparator_subj, KL_comparator_subj = self.comparator_distrib(normed_data['comparator'], x_comparator, var_comparator)

            # compute community distributions
            # combine subject distributions for each group, reshape: GxKxS -> KxGxS
            x = torch.permute(torch.stack([x_pre_perturb_subj, x_comparator_subj, x_post_perturb_subj]), (1,0,2))
            # group mean distribution
            x_mean = torch.stack([x_pre_perturb, x_comparator, x_post_perturb]).T # final shape = KxG
        else:
            KL_delta_comparator = KL_c_comparator = KL_var_comparator = KL_comparator_subj = 0
            x = torch.permute(torch.stack([x_pre_perturb_subj, x_post_perturb_subj]), (1,0,2))
            # group mean distribution
            x_mean = torch.stack([x_pre_perturb, x_post_perturb]).T # final shape = KxG

        # sample sparsity indicators
        gamma, KL_gamma = self.sparsity_gamma()
        # compute community distribution
        beta = self.sparse_softmax(x, gamma)
        
        # store parameters
        self.beta_mean = self.sparse_softmax(x_mean[:,:,None], gamma)
        self.beta = beta
        self.perturbation_magnitude = delta
        self.perturbation_indicators = self.perturb_indicators.q_probs
        self.sparsity_probs = self.sparsity_gamma.q_probs
        self.mean_pre_perturbation_distribution = x_pre_perturb
        self.mean_post_perturbation_distribution = x_post_perturb
        self.var_pre_perturbation = var_pre_perturb
        self.var_post_perturbation = var_post_perturb
        self.KL_pre_perturb = KL_pre_perturb
        self.KL_delta = KL_delta
        self.KL_c = KL_c
        self.KL_var_pre_perturb = KL_var_pre_perturb
        self.KL_var_post_perturb = KL_var_post_perturb
        self.KL_pre_perturb_subj = KL_pre_perturb_subj
        self.KL_post_perturb_subj = KL_post_perturb_subj
        self.KL_gamma = KL_gamma
        
        if self.num_groups == 3:
            self.comparator_perturbation_magnitude = delta_comparator #*
            self.comparator_perturbation_indicators = self.perturb_indicators_comparator.q_probs #*
            self.mean_comparator_distribution = x_comparator
            self.var_comparator = var_comparator
            self.KL_delta_comparator = KL_delta_comparator
            self.KL_c_comparator = KL_c_comparator
            self.KL_var_comparator = KL_var_comparator
            self.KL_comparator_subj = KL_comparator_subj
        else:
            self.comparator_perturbation_magnitude = None
            self.comparator_perturbation_indicators = None
            self.mean_comparator_distribution = None
            self.var_comparator = None
            self.KL_comparator = None
            self.KL_var_comparator = None
            self.KL_comparator_subj = None

        # return distribution and KL
        KL = KL_pre_perturb + KL_delta_comparator + KL_c_comparator + KL_delta + KL_c +\
                KL_var_pre_perturb + KL_var_comparator + KL_var_post_perturb +\
                KL_pre_perturb_subj + KL_comparator_subj + KL_post_perturb_subj + KL_gamma
        return beta, KL, gamma
    

class TimeSeriesCommunityDistribution(nn.Module):
    def __init__(self, num_communities, num_otus, num_time, perturbation_prior_prob, device, sparse_communities, scale_multiplier=1):
        super().__init__()
        self.num_communities = num_communities
        self.num_otus = num_otus
        self.device = device
        self.sparse_communities = sparse_communities
        self.sparsity_gamma = SparsityIndicator(num_communities, scale_multiplier, start_temp=0.5, end_temp=0.01, device=device)

        self.num_time = num_time
        self.perturbation_prior_prob = perturbation_prior_prob

        #* initial time point
        self.initial_distribution = InitialDistribution(num_communities, num_otus, device)

        #* perturbation parameters
        self.delta = TimeSeriesPerturbationMagnitude(num_communities, num_time-1, num_otus, device)
        self.perturb_indicators = PerturbationIndicators(shape=(num_communities,num_time-1),
                                                         prior_prob=perturbation_prior_prob,
                                                         start_temp=0.5,
                                                         end_temp=0.01,
                                                         device=device)
    
    def set_temps(self, epoch, num_epochs):
        self.sparsity_gamma.set_temp(epoch, num_epochs)
        self.perturb_indicators.set_temp(epoch, num_epochs)

    def get_params(self):
        return {
            'beta': self.beta,
            'beta_mean': self.beta,
            'x_initial': self.x_initial,
            'sparsity_probs': self.sparsity_probs,
            'perturbation_indicators': self.perturbation_indicators,
            'perturbation_magnitude': self.perturbation_magnitudes,
            'KL_x_initial': self.KL_x_initial,
            'KL_c': self.KL_c,
            'KL_delta': self.KL_delta,
            'KL_gamma': self.KL_gamma
        }

    def sparse_softmax(self, x, gamma):
        # a = torch.amax(x,dim=0)
        temp = gamma[:,None]*torch.exp(x) # - a)
        res = temp/temp.sum(dim=0)
        if (torch.isnan(res).any()) or (torch.isinf(res).any()):
            raise ValueError("nan or inf in sparse softmax")
        return res 

    def forward(self, inputs):
        normed_data = inputs['normed_data']

        # sample initial distribution
        x_initial, KL_x_initial = self.initial_distribution(normed_data['TS1'])

        # sample perturbation magnitudes and indicators
        delta, KL_delta = self.delta(inputs)
        c_indicators, KL_c = self.perturb_indicators()

        # compute time-series distribution
        x_kt = torch.zeros((self.num_communities, self.num_time)).to(self.device)
        x_kt[:,0] = x_initial
        for t in range(1,self.num_time):
            x_kt[:,t] = x_kt[:,t-1] + delta[:,t-1]*c_indicators[:,t-1]

        # sample sparsity indicators
        gamma, KL_gamma = self.sparsity_gamma()

        # compute community distributions
        beta = self.sparse_softmax(x_kt, gamma)

        # store parameters
        self.beta = beta
        self.x_initial = x_initial
        self.sparsity_probs = self.sparsity_gamma.q_probs
        # self.perturbation_indicators = c_indicators
        self.perturbation_indicators = self.perturb_indicators.q_probs
        self.perturbation_magnitudes = delta
        self.KL_x_initial = KL_x_initial
        self.KL_c = KL_c
        self.KL_delta = KL_delta
        self.KL_gamma = KL_gamma

        # return distribution and KL
        KL = KL_x_initial + KL_c + KL_delta + KL_gamma
        return beta, KL, gamma
        