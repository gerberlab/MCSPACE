import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax
from mcspace.utils import GaussianKL, BernoulliKL, BernoulliKLPower, inverse_softplus, sparse_softmax


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


class PerturbationIndicators(AnnealedParameter):
    def __init__(self, num_assemblages, num_perturbations, prior_prob, start_temp, end_temp, device):
        super().__init__(start_temp=start_temp, end_temp=end_temp) 
        self.num_assemblages = num_assemblages
        self.num_perturbations = num_perturbations
        self.device = device 
        self.gamma = None 
        self.prior_prob = torch.tensor(prior_prob).to(dtype=torch.float, device=self.device)
        q_gamma_params = torch.zeros((num_assemblages, num_perturbations), requires_grad=True, device=device)
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
        KL_k = BernoulliKL(self.q_gamma_params, self.prior_prob)
        KL = KL_k.sum()
        return self.gamma, KL 
    

class PerturbationMagnitude(nn.Module):
    def __init__(self, num_assemblages, num_perturbations, num_otus, device, prior_mean=0, prior_var=100):
        super().__init__()
        self.num_assemblages = num_assemblages
        self.num_perturbations = num_perturbations
        self.num_otus = num_otus
        self.hidden_dim = 10 #50
        self.device = device
        self.prior_mean = torch.tensor(prior_mean).to(device)
        self.prior_var = torch.tensor(prior_var).to(device)

        self.q_encode = nn.Sequential(
            nn.Linear(num_otus, self.hidden_dim),
            nn.Softplus(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Softplus()
        )
        self.q_mu_params = nn.Linear(self.hidden_dim, self.num_assemblages*self.num_perturbations)
        self.q_var_params = nn.Linear(self.hidden_dim, self.num_assemblages*self.num_perturbations)
        self.eps = None

    def sample_eps(self):
        self.eps = torch.normal(0,1,size=(self.num_assemblages,self.num_perturbations),device=self.device,requires_grad=False) 

    def forward(self, input):
        self.sample_eps()
        data = input['full_normed_data']
        enc = self.q_encode(data).mean(dim=0)
        mu = self.q_mu_params(enc).view(self.num_assemblages,self.num_perturbations)
        var = torch.exp(self.q_var_params(enc)).view(self.num_assemblages,self.num_perturbations)
        x = mu + torch.sqrt(var)*self.eps 
        KL = GaussianKL(mu, var, self.prior_mean, self.prior_var).sum()
        return x, KL
    

#! per subject process variance
#! ONLY if number of non-perturbed time points >2 ...
class ProcessVariance(nn.Module):
    def __init__(self, num_otus, times, subjects, prior_mean, prior_var, device):
        super().__init__()
        self.num_otus = num_otus
        self.subjects = subjects
        self.num_subjects = len(subjects)
        self.times = times
        self.num_time = len(times)
        #* invert softplus
        self.prior_mean = torch.tensor(inverse_softplus(prior_mean)).to(device)
        self.prior_var = torch.tensor(prior_var).to(device)
        self.device = device
        self.hidden_dim = 10

        self.q_encode = nn.Sequential(
            nn.Linear(num_otus, self.hidden_dim),
            nn.Softplus(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Softplus()
        )

        self.q_mu_params = nn.Linear(self.hidden_dim, 1)
        self.q_var_params = nn.Linear(self.hidden_dim, 1)
        self.eps = None 

    def sample_eps(self):
        self.eps = torch.normal(0, 1, size=(self.num_subjects,), device=self.device, requires_grad=False)

    def forward(self, input):
        self.sample_eps()
        data = input['normed_data']
        enc = torch.zeros(self.num_subjects, self.hidden_dim).to(self.device)
        for s_ind, s in enumerate(self.subjects):
            temp = 0
            for tm in self.times:
                temp += self.q_encode(data[tm][s]).mean(dim=0)
            enc[s_ind,:] = temp/self.num_time

        mu = torch.squeeze(self.q_mu_params(enc))
        var = torch.squeeze(torch.exp(self.q_var_params(enc)))
        x = mu + torch.sqrt(var)*self.eps 
        x = F.softplus(x)
        KL = GaussianKL(mu, var, self.prior_mean, self.prior_var)
        return x, KL
    

#! the "x's"
class LatentTimeSeriesMixtureWeights(nn.Module):
    def __init__(self, num_assemblages, num_otus, times, subjects, perturbed_times, device):
        super().__init__()
        self.perturbed_times = perturbed_times
        self.num_assemblages = num_assemblages
        self.num_otus = num_otus
        self.times = times
        self.subjects = subjects
        self.num_time = len(times)
        self.num_subjects = len(subjects)
        self.device = device

        self.hidden_dim = 50 
        self.q_encode = nn.Sequential(
            nn.Linear(num_otus, self.hidden_dim),
            nn.Softplus(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Softplus()
        )
        self.q_mu_params = nn.Linear(self.hidden_dim, self.num_assemblages)
        self.q_var_params = nn.Linear(self.hidden_dim, self.num_assemblages)
        self.eps = None

    def sample_eps(self):
        self.eps = torch.normal(0, 1, size=(self.num_assemblages, self.num_time, self.num_subjects), device=self.device, requires_grad=False)

    def compute_KL(self, mu, var, x_latent, delta, c_indicators, var_process):
        #* KL for initial time point
        KL_latent = 0.5*torch.sum(var[:,0,:] + (mu[:,0,:]**2) - 1.0 - torch.log(var[:,0,:]))

        #* compute KL for later time points
        p = 0
        for t in range(1,self.num_time):
            if self.perturbed_times[t] == 1:
                eta = x_latent[:,t-1,:] + c_indicators[:,p,None]*delta[:,p,None]
                p += 1
            else:
                eta = x_latent[:,t-1,:] - c_indicators[:,p-1,None]*delta[:,p-1,None]
            dt = self.times[t]-self.times[t-1]

            logq = torch.distributions.Normal(loc=mu[:,t,:], scale=torch.sqrt(var[:,t,:])).log_prob(x_latent[:,t,:]).sum()
            logp = torch.distributions.Normal(loc=eta, scale=torch.sqrt(dt*var_process[None,None,:])).log_prob(x_latent[:,t,:]).sum()
            KL_latent += (logq - logp)
        return KL_latent

    def forward(self, input, delta, c_indicators, var_process=None):
        self.sample_eps()
        data = input['normed_data']

        enc = torch.zeros(self.num_time, self.num_subjects, self.hidden_dim).to(self.device)
        for t_ind, tm in enumerate(self.times):
            for s_ind, s in enumerate(self.subjects):
                normed_reads = data[tm][s]
                enc[t_ind,s_ind,:] = self.q_encode(normed_reads).mean(dim=0) # mean over particles in given sample
        enc = torch.reshape(enc, (self.num_time*self.num_subjects, self.hidden_dim))

        # evaluate posterior mean and variance params; reshape to NxSxK; permute to KxNxS
        mu = torch.reshape(self.q_mu_params(enc).T, (self.num_assemblages, self.num_time, self.num_subjects))
        var = torch.reshape(torch.exp(self.q_var_params(enc)).T, (self.num_assemblages, self.num_time, self.num_subjects))

        # shape = kxtxs
        x_latent = mu + torch.sqrt(var)*self.eps 

        if var_process is not None:
            KL_latent = self.compute_KL(mu, var, x_latent, delta, c_indicators, var_process)
        elif self.num_time == 2:
            #! if no process var, use first and compute rest deterministically...
            x_latent[:,1,:] = x_latent[:,0,:] + c_indicators[:,0,None]*delta[:,0,None]
            KL_latent = GaussianKL(mu[:,0,:], var[:,0,:], torch.tensor(0).to(self.device), torch.tensor(1).to(self.device)).sum()
        elif self.num_time == 1:
            KL_latent = GaussianKL(mu[:,0,:], var[:,0,:], torch.tensor(0).to(self.device), torch.tensor(1).to(self.device)).sum()

        return x_latent, KL_latent



class SparsityIndicator(AnnealedParameter):
    def __init__(self, num_assemblages, prior_prob, device,
                 start_temp=0.5, end_temp=0.01, KL_scale_multiplier=1):
        super().__init__(start_temp, end_temp)
        # first indicator is fixed to 1; sampling only (num_assemblages - 1) components
        self.num_assemblages = num_assemblages-1 
        self.device = device

        # prior probability
        self.prior_prob = torch.tensor(prior_prob).to(self.device)

        # latent logit params
        q_gamma_params = torch.zeros((self.num_assemblages,), requires_grad=True, device=device)
        torch.nn.init.uniform_(q_gamma_params)
        self.q_gamma_params = torch.nn.Parameter(q_gamma_params)
        
        # posterior probabilities of indicators (= sigmoid(self.q_gamma_params))
        self.q_probs = None
        self.gamma = None

        # multiplier for KL
        self.KL_scale_multiplier = KL_scale_multiplier
        self.KL_gamma_k = None
        self.KL_gamma = None
        self.KL = None

    def sample_gamma(self):
        log_prob = F.logsigmoid(torch.stack((self.q_gamma_params, -self.q_gamma_params)))
        gamma_rest = gumbel_softmax(log_prob, hard=True, dim=0, tau=self.concrete_temperature)[0]
        self.gamma = torch.cat((torch.tensor([1.0]).to(self.device), gamma_rest))
        if torch.isnan(self.gamma).any():
            raise ValueError("nan in gamma")
        
    def forward(self):
        self.sample_gamma()
        q = torch.sigmoid(self.q_gamma_params)
        self.q_probs = q
        self.KL_gamma_k = BernoulliKL(self.q_gamma_params, self.prior_prob)
        self.KL_gamma = self.KL_gamma_k.sum()
        # rescale with multiplier
        self.KL = self.KL_scale_multiplier*self.KL_gamma
        return self.gamma, self.KL
    

class SparsityIndicatorPowerScale(SparsityIndicator):
    def __init__(self, num_assemblages, prior_prob, prob_power, device,
                 start_temp=0.5, end_temp=0.01):
        super().__init__(num_assemblages, prior_prob, device,
                 start_temp, end_temp, KL_scale_multiplier=1)
        self.log_prior = torch.log(self.prior_prob) #.to(self.device)
        self.prior_power = prob_power

    def forward(self):
        self.sample_gamma()
        q = torch.sigmoid(self.q_gamma_params)
        self.q_probs = q
        self.KL_gamma_k = BernoulliKLPower(self.q_gamma_params, self.log_prior, self.prior_power)
        KL= self.KL_gamma_k.sum()
        return self.gamma, KL


class AssemblageProportions(nn.Module):
    def __init__(self,
                num_assemblages,
                num_otus,
                times,
                subjects,
                perturbed_times,
                sparsity_prior,
                sparsity_prior_power,
                process_var_prior_mean,
                perturbation_prior,
                device,
                use_sparse_weights,
                add_process_variance):
        super().__init__()

        self.num_assemblages = num_assemblages 
        self.num_otus = num_otus 
        self.times = times
        self.num_time = len(times)
        self.subjects = subjects
        self.device = device
        self.add_process_variance = add_process_variance

        self.perturbed_times = perturbed_times
        self.num_perturbations = np.array(perturbed_times).sum() #! since input is boolean...
        self.perturbation_prior_prob = perturbation_prior

        self.use_sparse_weights = use_sparse_weights
        if use_sparse_weights is True:
            self.sparsity_params = SparsityIndicatorPowerScale(num_assemblages, 
                                                           sparsity_prior, 
                                                           sparsity_prior_power,
                                                           device)
        else:
            self.gamma_ones = torch.ones(self.num_assemblages).to(device)

        if self.num_perturbations > 0:
            #* perturbation indicators
            self.perturbation_indicators = PerturbationIndicators(num_assemblages, self.num_perturbations, self.perturbation_prior_prob, 
                                                                    start_temp=0.5, end_temp=0.01, device=device)            
            #* perturbation magnitude
            self.perturbation_magnitude = PerturbationMagnitude(num_assemblages, self.num_perturbations, num_otus, device, prior_mean=0, prior_var=100)

        if self.add_process_variance is True:
            self.process_var = ProcessVariance(num_otus, times, subjects, process_var_prior_mean, prior_var=10, device=device)

        self.latent_distrib = LatentTimeSeriesMixtureWeights(num_assemblages,
                                                             num_otus,
                                                             times,
                                                             subjects,
                                                             perturbed_times,
                                                             device)

    def forward(self, data):
        # if perturbations
        if self.num_perturbations > 0:
            delta, KL_delta = self.perturbation_magnitude(data)
            c_indicators, KL_c = self.perturbation_indicators()
        else:
            delta = c_indicators = torch.zeros((self.num_assemblages, self.num_time))
            KL_delta = KL_c = 0
        
        # if time series
        if self.add_process_variance:
            var_process, KL_var_process = self.process_var(data)
        else:
            var_process = None
            KL_var_process = 0
        
        # if sparse
        if self.use_sparse_weights:
            gamma, KL_gamma = self.sparsity_params()
        else:
            gamma = self.gamma_ones
            KL_gamma = 0

        x_latent, KL_x_latent = self.latent_distrib(data, delta, c_indicators, var_process)

        beta = sparse_softmax(x_latent, gamma)

        KL = KL_delta + KL_c + KL_var_process + KL_gamma + KL_x_latent
        return beta, KL, gamma
    