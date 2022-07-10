import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 


def visualize_particle_locations(locs, **kwargs):
    npart, ndim = locs.shape

    if ndim == 2:
        fig, ax = plt.subplots() 
        ax.scatter(locs[:,0], locs[:,1])
        return fig, ax
    else:
        print("not implemented")


def visualize_particle_reads(reads, **kwargs):
    """
    reads: numpy array of shape (number particles) x (number otus) giving counts
    """
    fig, ax = plt.subplots()
    ax = sns.heatmap(data=reads.T, cmap='Greys', ax=ax)  
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    return ax 


def visualize_particle_relative_abundances(reads, ax=None, **kwargs):
    """
    reads: numpy array of shape (number particles) x (number otus) giving counts
    """
    # calculate relative abundance
    ra = reads/(np.sum(reads, axis=1, keepdims=True))
    # sort by composition for otu

    ax = sns.heatmap(data=ra.T, ax=ax, cmap='Greys', cbar_kws = dict(use_gridspec=False,location="bottom"))  
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax 
