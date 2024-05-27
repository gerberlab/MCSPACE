import numpy as np
import torch
from torch.distributions.beta import Beta
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from mcspace.utils import inverse_softplus

#! depending on data, may want to model for each time point (averaged over subjects) instead?...

#! gen data with known contamination and test....

#* pi ~ Beta(b1, b2)
#* prior = Beta(0.05, 1.0)
class GarbageWeights(nn.Module):
    def __init__(self, num_time, num_subjects, num_otus, device):
        super().__init__()
        self.num_time = num_time
        self.num_subjects = num_subjects
        self.num_otus = num_otus
        self.device = device

        prior_b1 = 0.05
        prior_b2 = 1.0
        self.prior_b1 = torch.tensor(prior_b1, requires_grad=False, device=self.device)
        self.prior_b2 = torch.tensor(prior_b2, requires_grad=False, device=self.device)
        self.prior_dist = Beta(self.prior_b1, self.prior_b2)

        #! todo - add encoder

        # self.b1_params = nn.Parameter(torch.normal(inverse_softplus(prior_b1), 0.1, size=(self.num_time, self.num_subjects)), requires_grad=True)
        # self.b2_params = nn.Parameter(torch.normal(inverse_softplus(prior_b2), 0.1, size=(self.num_time, self.num_subjects)), requires_grad=True)
        #! intializing close to prior seems like its getting stuck near prior...
        self.b1_params = nn.Parameter(torch.normal(0, 1, size=(self.num_time, self.num_subjects)), requires_grad=True)
        self.b2_params = nn.Parameter(torch.normal(0, 1, size=(self.num_time, self.num_subjects)), requires_grad=True)

    def forward(self, data):
        # need to take softplus of these for valid params
        b1 = F.softplus(self.b1_params)
        b2 = F.softplus(self.b2_params)

        beta = Beta(b1, b2)
        pi = beta.rsample()
        KL = kl_divergence(beta, self.prior_dist).sum()
        return pi, KL


#! test garbage weight class with normal prior? (compare cv? -- setup pipeline and run on cluster)
#* as well as synthetic results...


#! test optimizing to prior?
class TestGarbageWeights(nn.Module):
    def __init__(self, num_time, num_subjects, device, lr=0.01):
        super().__init__()

        self.lr = lr

        self.gweights = GarbageWeights(num_time, num_subjects, num_otus=1, device=device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def get_params(self):
        params = {
            'b1_params': F.softplus(self.gweights.b1_params),
            'b2_params': F.softplus(self.gweights.b2_params),
        }
        return params
    
    def forward(self):
        pi, KL = self.gweights()
        self.ELBO_loss = KL
        return self.ELBO_loss, pi


#! test optimizing to objective, what does posterior look like?
class TestObjective(nn.Module):
    def __init__(self, num_time, num_subjects, pi_mean, pi_var, device, lr=0.01):
        super().__init__()

        self.pi_mean = torch.tensor(pi_mean, device=device, requires_grad=False)
        self.pi_var = torch.tensor(pi_var, device=device, requires_grad=False)

        self.lr = lr

        self.gweights = GarbageWeights(num_time, num_subjects, num_otus=1, device=device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def get_params(self):
        params = {
            'b1_params': F.softplus(self.gweights.b1_params),
            'b2_params': F.softplus(self.gweights.b2_params),
        }
        return params
    
    def forward(self, data):
        pi, KL = self.gweights(data)
        loglik = torch.distributions.normal.Normal(self.pi_mean, torch.sqrt(self.pi_var)).log_prob(pi).sum()
        self.ELBO_loss = -(loglik - KL)
        return self.ELBO_loss, pi



def train(model, data, num_epochs, verbose=True):
    model.train() 
    ELBOs = np.zeros(num_epochs) 

    for epoch in range(0, num_epochs):
        model.forward(data)
        model.optimizer.zero_grad() 
        model.ELBO_loss.backward() 
        model.optimizer.step() 
        if verbose:
            if epoch % 10 == 0:
                print(f"\nepoch {epoch}")
                print("ELBO = ", model.ELBO_loss)
        ELBOs[epoch] = model.ELBO_loss.cpu().clone().detach().numpy() 
    return ELBOs 
