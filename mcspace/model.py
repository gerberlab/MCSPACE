import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from mcspace.assemblage_proportions import AssemblageProportions
from mcspace.utils import ilr_transform_data, inv_ilr_transform_data


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
                lr=5e-3):
        super().__init__()

        self.num_assemblages = num_assemblages 
        self.num_otus = num_otus 
        self.times = times
        self.num_time = len(times)
        self.subjects = subjects
        self.device = device
        self.lr = lr

        self.add_process_variance = add_process_variance

        self.perturbed_times = perturbed_times
        self.num_perturbations = (np.array(perturbed_times)==1).sum()
        self.perturbation_prior_prob = perturbation_prior

        self.use_sparse_weights = use_sparse_weights

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
                                                 add_process_variance)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def init_theta_kmeans(self, data):
        # counts = data['count_data'].cpu().detach().clone().numpy()
        reads = data['count_data']
        times = list(reads.keys())
        subjs = list(reads[times[0]].keys())
        subj_counts = []
        for t in times:
            for s in subjs:
                subj_counts.append(reads[t][s].cpu().detach().clone().numpy())
        counts = np.concatenate(subj_counts, axis=0)

        ilr_reads = ilr_transform_data(counts)
        model = KMeans(n_clusters=self.num_assemblages)
        km_labels = model.fit_predict(ilr_reads)
        kmeans_theta = inv_ilr_transform_data(model.cluster_centers_)
        theta_init = np.log(kmeans_theta)
        self.theta_params.data = torch.from_numpy(theta_init).to(self.device)
        print("theta intialized from kmeans fit")

    def set_temps(self, epoch, num_epochs):
        if self.use_sparse_weights:
            self.beta_params.sparsity_params.set_temp(epoch, num_epochs)
        if self.num_perturbations > 0:
            self.beta_params.perturbation_indicators.set_temp(epoch, num_epochs)

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
    
    def forward(self, data):
        counts = data['count_data'] # is dict over times and subjects

        beta, KL, gamma = self.beta_params(data)
        theta = F.softmax(self.theta_params, dim=1)
        loglik = self.compute_loglik_multinomial(beta, theta, counts)

        self.ELBO_loss = -(loglik - KL)
        return self.ELBO_loss, theta, beta, gamma
