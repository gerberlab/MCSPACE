import torch
import numpy as np


def train_model(model, data, num_epochs, anneal_prior=True, verbose=True):
    model.train() 
    ELBOs = np.zeros(num_epochs) 

    for epoch in range(0, num_epochs):
        model.set_temps(epoch, num_epochs)
        if anneal_prior:
            model.anneal_gamma_prior(epoch, num_epochs)
        model.forward(data)
        model.optimizer.zero_grad() 
        model.ELBO_loss.backward() 
        model.optimizer.step() 
        if verbose:
            if epoch % 100 == 0:
                print(f"\nepoch {epoch}")
                print("ELBO = ", model.ELBO_loss)
        ELBOs[epoch] = model.ELBO_loss.cpu().clone().detach().numpy() 
    return ELBOs 
