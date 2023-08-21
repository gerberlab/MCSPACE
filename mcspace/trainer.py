import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt 
import time
from mcspace.utils import RESULT_FILE, MODEL_FILE, DATA_FILE, pickle_save, move_to_numpy, save_model
# from mcspace.models import BasicModel, PerturbationModel, TSModel
# from mcspace.models import PerturbationModel


def train_model(model, data, num_epochs, verbose):
    model.train() 
    ELBOs = np.zeros(num_epochs) 

    for epoch in range(0, num_epochs):
        model.community_distribution.set_temps(epoch, num_epochs)
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


def train(model, data, num_epochs, outpath, verbose=True):
    st = time.time() 

    model.community_distribution.set_gamma_scale() #data)
    ELBOs = train_model(model, data, num_epochs, verbose=True)
    model.eval() # TODO: should be no different, take samples...
    loss, theta, beta, pi_garb = model(data)
    params = model.community_distribution.get_params()

    #* save results; plot in separate file 
    results = {'theta': theta, 'beta': beta, 'params': params, 'loss': loss, 'ELBOs': ELBOs, 'pi_garb': pi_garb}
    pickle_save(results, outpath / RESULT_FILE, to_numpy=True)
    pickle_save(data, outpath / DATA_FILE, to_numpy=False)
    save_model(model, outpath / MODEL_FILE)

    # plot losses
    fig, ax = plt.subplots()
    ax.plot(ELBOs)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ELBO loss")
    plt.savefig(outpath / "ELBO_loss.png")
    plt.close()

    #* get run time
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
    print("***ALL DONE***")

