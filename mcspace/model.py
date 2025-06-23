import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mcspace.assemblage_proportions import AssemblageProportions
from mcspace.garbage_weights import ContaminationWeights
from mcspace.utils import ilr_transform_data, inv_ilr_transform_data
from sklearn.cluster import KMeans


def flatten_data(reads):
    times = list(reads.keys())
    subjs = list(reads[times[0]].keys())
    combreads = []
    for t in times:
        for s in subjs:
            combreads.append(reads[t][s].cpu().detach().clone().numpy())
    allreads = np.concatenate(combreads, axis=0)    
    return allreads


class MCSPACE(nn.Module):
    def __init__(self,
                num_assemblages,
                num_otus,
                times,
                subjects,
                perturbed_times,
                perturbation_prior,
                sparsity_prior,
                sparsity_prior_power,
                process_var_prior_mean,
                device,
                add_process_variance,
                use_sparse_weights=True,
                use_contamination=False,
                contamination_clusters=None,
                lr=5e-3,
                process_var_prior_scale=10,
                perturbation_magnitude_prior_scale=100,
                garbage_prior_scale=10):
        super().__init__()

        self.num_assemblages = num_assemblages 
        self.num_otus = num_otus 
        self.times = times
        self.num_time = len(times)
        self.subjects = subjects
        self.num_subjects = len(subjects)
        self.device = device
        self.lr = lr

        self.add_process_variance = add_process_variance

        self.perturbed_times = perturbed_times
        self.num_perturbations = (np.array(perturbed_times)==1).sum()
        self.perturbation_prior_prob = perturbation_prior

        self.use_sparse_weights = use_sparse_weights

        # mixing contamination
        self.use_contamination = use_contamination
        self.contamination_clusters = contamination_clusters

        self.garbage_weights = ContaminationWeights(self.num_time, self.device, prior_var=garbage_prior_scale)

        self.theta_params = nn.Parameter(torch.normal(0, 1, size=(self.num_assemblages, self.num_otus), device=self.device, requires_grad=True))
        self.beta_params = AssemblageProportions(num_assemblages,
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
                                                 add_process_variance,
                                                 process_var_prior_scale,
                                                 perturbation_magnitude_prior_scale)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)

    def kmeans_init(self, data, seed=0):
        counts = flatten_data(data['count_data'])
        ilr_data = ilr_transform_data(counts)
        kmodel = KMeans(n_clusters=self.num_assemblages, random_state=seed)
        res = kmodel.fit(ilr_data)
        clusters = np.log(inv_ilr_transform_data(res.cluster_centers_))
        clusters -= np.mean(clusters) #! allowed to invert softmax by constant; this centers it...
        self.theta_params.data = torch.from_numpy(clusters).float()

    def set_temps(self, epoch, num_epochs):
        if self.use_sparse_weights:
            self.beta_params.sparsity_params.set_temp(epoch, num_epochs)
        if self.num_perturbations > 0:
            self.beta_params.perturbation_indicators.set_temp(epoch, num_epochs)

    #! if not called, should stick to default full value
    def anneal_gamma_prior(self, epoch, num_epochs):
        self.beta_params.sparsity_params.anneal_prior_power(epoch, num_epochs)

    def compute_loglik_multinomial(self, beta, theta, counts):
        EPS = 1e-6
        total = 0
        for i, tm in enumerate(self.times):
            for j, sub in enumerate(self.subjects): 
                rlogtheta = torch.sum(counts[tm][sub][:,None,:]*torch.log(theta + EPS), dim=2)
                x_star, _ = torch.max(rlogtheta, dim=1, keepdim=True)
                summand = x_star + torch.log(torch.sum(beta[None,:,i,j]*torch.exp(rlogtheta - x_star) ,dim=1,keepdim=True) + EPS)
                total += summand.sum()
        if torch.isnan(total).any():
            raise ValueError("nan in sparse loglik")
        if torch.isinf(total).any():
            raise ValueError("inf in sparse loglik")
        return total

    def compute_loglik_garbage(self, beta, theta, counts, gweights):
        EPS = 1e-6
        total = 0
        for i, tm in enumerate(self.times):
            for j, sub in enumerate(self.subjects): 
                pi = gweights[i]
                garb_cluster = self.contamination_clusters[tm]
                theta_mix = (1.0 - pi)*theta + pi*garb_cluster[None,:] # KxO
                rlogtheta = torch.sum(counts[tm][sub][:,None,:]*torch.log(theta_mix + EPS), dim=2)
                x_star, _ = torch.max(rlogtheta, dim=1, keepdim=True)
                summand = x_star + torch.log(torch.sum(beta[None,:,i,j]*torch.exp(rlogtheta - x_star) ,dim=1,keepdim=True) + EPS)
                total += summand.sum()
        if torch.isnan(total).any():
            raise ValueError("nan in sparse loglik")
        if torch.isinf(total).any():
            raise ValueError("inf in sparse loglik")
        return total

    def forward(self, data):
        counts = data['count_data'] # dict over times and subjects

        beta, KL_beta, gamma = self.beta_params(data)
        theta = F.softmax(self.theta_params, dim=1)
    
        if self.use_contamination:
            pi, KL_pi = self.garbage_weights()
            loglik = self.compute_loglik_garbage(beta, theta, counts, pi)
        else:
            loglik = self.compute_loglik_multinomial(beta, theta, counts)
            pi = None            
            KL_pi = 0

        KL = KL_beta + KL_pi
        self.ELBO_loss = -(loglik - KL)
        self.loglik = loglik
        return self.ELBO_loss, theta, beta, gamma, pi
