import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax
from mcspace.otu_distributions import BasicOtuDistribution
from mcspace.community_distributions import BasicCommunityDistribution, PerturbationCommunityDistribution, TimeSeriesCommunityDistribution

EPS = 1e-6 #10

def sparse_logsumexp(x, gamma, dim=1):
    # x shape LxK
    x_star, _ = torch.max(x, dim=dim, keepdim=True) 
    #torch.max(x[:,gamma>0.5], dim=dim, keepdim=True) #! should be biggest contributor to LL*** (max operation of K for each L)
    # TODO: prob want to take max of non-zeroed only...
    y = x_star + (((gamma*(x - x_star)).exp())).sum(dim=dim, keepdim=True).log()

    # y = x_star + torch.log(torch.sum(gamma*torch.exp(gamma*(x-x_star)), dim=dim,keepdim=True))

    if torch.isnan(y).any():
        raise ValueError("nan in sparse LSE")
    if torch.isinf(y).any():
        raise ValueError("inf in sparse LSE")
    return y

#* to test between deterministic or stochastic parameter...
# TODO: move to otu_distributions file instead
# TODO: give shape, option for stochastic vs deterministic, and prior settings...
class ContaminationWeights(nn.Module):
    def __init__(self, num_groups, device, stochastic=True):
        super().__init__()
        self.num_groups = num_groups
        self.stochastic = stochastic
        self.device = device

        if self.stochastic is True:
            self.prior_mean = torch.logit(torch.tensor(0.05)) # so expected value is around 5%
            self.prior_var = 10 # diffuse prior
            self.q_mu = nn.Parameter(torch.normal(self.prior_mean, 1, size=(num_groups,)), requires_grad=True)
            self.q_var = nn.Parameter(torch.normal(0, 1, size=(num_groups,)), requires_grad=True)
        else:
            self.weights = nn.Parameter(torch.normal(0, 1, size=(num_groups,)), requires_grad=True)
    
    def sample_eps(self):
        self.eps = torch.normal(0, 1, size=(self.num_groups,), device=self.device, requires_grad=False)

    def forward(self):
        if self.stochastic is True:
            self.sample_eps()
            mu = self.q_mu
            var = torch.exp(self.q_var)
            x = mu + torch.sqrt(var)*self.eps
            pi_weight = torch.sigmoid(x)
            KL = 0.5*torch.sum(var/self.prior_var + ((mu-self.prior_mean)**2)/self.prior_var - 1.0 - torch.log(var/self.prior_var))
        else:
            pi_weight = torch.sigmoid(self.weights)
            KL = 0
        return pi_weight, KL
    

class BasicModel(nn.Module):
    def __init__(self,
                 num_communities,
                 num_otus,
                 device,
                 sparse_communities=True,
                 use_contamination_community=False,
                 contamination_community=None,
                 lr=1e-3,
                 KL_scale_beta=1):
        super().__init__()

        self.num_communities = num_communities 
        self.num_otus = num_otus 
        self.lr = lr
        self.device = device 

        self.use_contamination_community = use_contamination_community 
        self.sparse_communities = sparse_communities

        # contamination communities
        if use_contamination_community is True:
            self.contamination_weight = ContaminationWeights(1, device)
            self.contamination_community = contamination_community
   
        self.otu_distribution = BasicOtuDistribution(num_communities, num_otus, device)
        self.community_distribution = BasicCommunityDistribution(num_communities, num_otus, device, sparse_communities, scale_multiplier=KL_scale_beta)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def compute_log_likelihood(self, comm_dist, otu_dist, counts, contam_weight, gamma):
        if self.use_contamination_community:
            otu_mixture_distrib = otu_dist*(1-contam_weight) + contam_weight*self.contamination_community
        else:
            otu_mixture_distrib = otu_dist
        logsumexpand = (counts[:,None,:]*torch.log(otu_mixture_distrib[None,:,:] + EPS)).sum(dim=-1) + torch.log(comm_dist[None,:] + EPS)
        
        summand = sparse_logsumexp(logsumexpand, gamma, dim=1)
        # summand = torch.logsumexp(logsumexpand, dim=1)
        return  summand.sum()

    def forward(self, inputs):
        count_data = inputs['count_data']
        comm_distrib, KL_comm, gamma = self.community_distribution(inputs) 
        otu_distrib, KL_otu = self.otu_distribution(inputs)

        if self.use_contamination_community is True:
            pi_contam, KL_contam = self.contamination_weight()
        else:
            pi_contam = None
            KL_contam = 0

        data_loglik = self.compute_log_likelihood(comm_distrib, otu_distrib, count_data, pi_contam, gamma)
        KL = KL_comm + KL_otu + KL_contam
        self.ELBO_loss = -(data_loglik - KL)

        return self.ELBO_loss, otu_distrib, comm_distrib, pi_contam, gamma


# class BasicModelSqrtData(BasicModel):
#     def __init__(self,
#                  num_communities,
#                  num_otus,
#                  device,
#                  sparse_communities=True,
#                  use_contamination_community=False,
#                  contamination_community=None,
#                  lr=1e-3):
#         super().__init__(num_communities,
#                         num_otus,
#                         device,
#                         sparse_communities,
#                         use_contamination_community,
#                         contamination_community,
#                         lr)

#     def compute_log_likelihood(self, comm_dist, otu_dist, counts, contam_weight):
#         if self.use_contamination_community:
#             otu_mixture_distrib = otu_dist*(1-contam_weight) + contam_weight*self.contamination_community
#         else:
#             otu_mixture_distrib = otu_dist
#         logsumexpand = (torch.sqrt(counts[:,None,:])*torch.log(otu_mixture_distrib[None,:,:] + EPS)).sum(dim=-1) + torch.log(comm_dist[None,:] + EPS)
#         summand = torch.logsumexp(logsumexpand, dim=1)
#         return  summand.sum()
    

class PerturbationModel(nn.Module):
    def __init__(self,
                 num_communities,
                 num_otus,
                 num_groups,
                 num_subjects,
                 subject_variance,
                 device,
                 sparse_communities=True,
                 scale_multiplier=1,
                 use_contamination_community=False,
                 contamination_community=None,
                 lr=1e-3):
        super().__init__()

        self.num_communities = num_communities 
        self.num_otus = num_otus 
        self.lr = lr
        self.device = device 

        self.num_groups = num_groups

        self.use_contamination_community = use_contamination_community 
        self.sparse_communities = sparse_communities

        # contamination communities
        if use_contamination_community is True:
            self.contamination_weight = ContaminationWeights(num_groups, device)
            self.contamination_community = contamination_community

        self.otu_distribution = BasicOtuDistribution(num_communities, num_otus, device)

        # TODO: add option for peturbation prior? or just keep at 'default'?
        perturbation_prior_prob = 0.5/num_communities
        self.community_distribution = PerturbationCommunityDistribution(num_communities, 
                                                                        num_otus, 
                                                                        num_groups,
                                                                        num_subjects, 
                                                                        subject_variance, 
                                                                        perturbation_prior_prob, 
                                                                        device, 
                                                                        sparse_communities,
                                                                        scale_multiplier=scale_multiplier)
    
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def compute_log_likelihood(self, comm_dist, otu_dist, counts, contam_weight, gamma):
        # count data is for multiple groups and subjects
        total = 0
        if self.num_groups == 3:
            groups = ['pre_perturb', 'comparator', 'post_perturb']
        else:
            groups = ['pre_perturb', 'post_perturb']
        for g_ind, grp in enumerate(groups):
            subjs = counts[grp].keys()
            for s_ind, s in enumerate(subjs):
                if self.use_contamination_community:
                    otu_mixture_distrib = otu_dist*(1-contam_weight[g_ind]) + contam_weight[g_ind]*self.contamination_community[grp]
                else:
                    otu_mixture_distrib = otu_dist
                logsumexpand = (counts[grp][s][:,None,:]*torch.log(otu_mixture_distrib[None,:,:] + EPS)).sum(dim=-1) + torch.log(comm_dist[None,:,g_ind,s_ind] + EPS)
                summand = sparse_logsumexp(logsumexpand, gamma, dim=1)
                # summand = torch.logsumexp(logsumexpand, dim=1) # logsumexp over communities K
                total += summand.sum() # sum over particles L

        if torch.isnan(total).any():
            raise ValueError("nan in sparse loglik")
        if torch.isinf(total).any():
            raise ValueError("inf in sparse loglik")
        return total

    def forward(self, inputs):
        count_data = inputs['count_data']
        comm_distrib, KL_comm, gamma = self.community_distribution(inputs) 
        otu_distrib, KL_otu = self.otu_distribution(inputs)

        if self.use_contamination_community is True:
            pi_contam, KL_contam = self.contamination_weight()
        else:
            pi_contam = None
            KL_contam = 0

        data_loglik = self.compute_log_likelihood(comm_distrib, otu_distrib, count_data, pi_contam, gamma)
        KL = KL_comm + KL_otu + KL_contam
        self.ELBO_loss = -(data_loglik - KL)

        return self.ELBO_loss, otu_distrib, comm_distrib, pi_contam, gamma
    

class TimeSeriesModel(nn.Module):
    def __init__(self,
                 num_communities,
                 num_otus,
                 num_time,
                 device,
                 sparse_communities=True,
                 scale_multiplier=1,
                 use_contamination_community=False,
                 contamination_community=None,
                 lr=1e-3):
        super().__init__()

        self.num_communities = num_communities 
        self.num_otus = num_otus 
        self.lr = lr
        self.device = device 
        
        self.use_contamination_community = use_contamination_community 
        self.sparse_communities = sparse_communities

        # contamination communities
        if use_contamination_community is True:
            self.contamination_weight = ContaminationWeights(1, device) #! note: using one one contamination comm for whole time series...
            self.contamination_community = contamination_community

        self.otu_distribution = BasicOtuDistribution(num_communities, num_otus, device)

        # TODO: add option for peturbation prior? or just keep at 'default'?
        perturbation_prior_prob = 0.5/(num_communities*(num_time-1))
        self.community_distribution = TimeSeriesCommunityDistribution(num_communities, 
                                                                        num_otus, 
                                                                        num_time,
                                                                        perturbation_prior_prob, 
                                                                        device, 
                                                                        sparse_communities,
                                                                        scale_multiplier=scale_multiplier)
    
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def compute_log_likelihood(self, comm_dist, otu_dist, counts, contam_weight, gamma):
        # count data is for multiple groups and subjects
        total = 0
        times = list(counts.keys())
        if self.use_contamination_community:
            otu_mixture_distrib = otu_dist*(1-contam_weight) + contam_weight*self.contamination_community
        else:
            otu_mixture_distrib = otu_dist
        for t_ind, tm in enumerate(times):   
            logsumexpand = (counts[tm][:,None,:]*torch.log(otu_mixture_distrib[None,:,:] + EPS)).sum(dim=-1) + torch.log(comm_dist[None,:,t_ind] + EPS)
            # summand = torch.logsumexp(logsumexpand, dim=1) # logsumexp over communities K
            summand = sparse_logsumexp(logsumexpand, gamma, dim=1)
            total += summand.sum() # sum over particles L
        return total

    def forward(self, inputs):
        count_data = inputs['count_data']
        comm_distrib, KL_comm, gamma = self.community_distribution(inputs) 
        otu_distrib, KL_otu = self.otu_distribution(inputs)

        if self.use_contamination_community is True:
            pi_contam, KL_contam = self.contamination_weight()
        else:
            pi_contam = None
            KL_contam = 0

        data_loglik = self.compute_log_likelihood(comm_distrib, otu_distrib, count_data, pi_contam, gamma)
        KL = KL_comm + KL_otu + KL_contam
        self.ELBO_loss = -(data_loglik - KL)

        return self.ELBO_loss, otu_distrib, comm_distrib, pi_contam, gamma
    