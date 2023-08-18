import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicOtuDistribution(nn.Module):
    def __init__(self, num_communities, num_otus, device):
        super().__init__()
        self.num_communities = num_communities 
        self.num_otus = num_otus 
        self.device = device 
        self.hidden_dim = 50
        self.q_otu_encode = nn.Sequential(
            nn.Linear(num_otus, self.hidden_dim),
            nn.Softplus(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Softplus()
        )
        self.q_otu_mu_params = nn.Linear(self.hidden_dim, self.hidden_dim) 
        self.q_otu_var_params = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.z_otu_params = nn.Linear(self.hidden_dim, self.num_otus*self.num_communities) 
        self.eps = None 
    
    def sample_eps(self):
        self.eps = torch.normal(0, 1, size=(self.hidden_dim,), device=self.device, requires_grad=False)

    def compute_otu_z(self, normed_data, samples):
        particle_otu_enc = self.q_otu_encode(normed_data) 
        enc = particle_otu_enc.mean(dim=0) 
        mu = self.q_otu_mu_params(enc) 
        var = torch.exp(self.q_otu_var_params(enc))
        z_otu = self.z_otu_params( (mu + torch.sqrt(var) * samples) ).view(self.num_communities, self.num_otus)
        return mu, var, z_otu

    def forward(self, inputs):
        normed_data = inputs['full_normed_data']
        self.sample_eps()
        mu_otu, var_otu, otu_z = self.compute_otu_z(normed_data, self.eps)
        otu_distrib = F.softmax(otu_z, dim=1)
        KL_otu = 0.5*torch.sum(var_otu + (mu_otu**2) - 1.0 - torch.log(var_otu))
        return otu_distrib, KL_otu
    