import numpy as np
import torch
from torch.distributions.beta import Beta
import torch.nn as nn
import torch.nn.functional as F


class ContaminationWeights(nn.Module):
    def __init__(self, num_groups, device, stochastic=True, prior_var=10):
        super().__init__()
        self.num_groups = num_groups
        self.stochastic = stochastic
        self.device = device

        if self.stochastic is True:
            self.prior_mean = torch.logit(torch.tensor(0.05)) # so expected value is around 5%
            self.prior_var = prior_var # diffuse prior
            self.q_mu = nn.Parameter(torch.normal(0, 1, size=(num_groups,)), requires_grad=True)
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
