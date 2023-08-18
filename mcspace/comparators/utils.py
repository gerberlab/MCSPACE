import numpy as np
from skbio.stats.composition import ilr, ilr_inv


def ilr_transform_data(data):
    EPS = 1e-8 #* below which data is considered to be '0'

    relative_abundance = data/(data.sum(axis=1, keepdims=True))
    
    #* deal with zeros
    nparticles, notus = relative_abundance.shape
    delta = 1.0/(notus**2) # imputed value
    
    zdata = np.zeros((nparticles, notus))
    for lidx in range(nparticles):
        zeros = (relative_abundance[lidx,:] < EPS)
        nzeros = zeros.sum()
        zdata[lidx,:] = (1.0 - nzeros*delta)*relative_abundance[lidx,:]
        zdata[lidx,zeros] = delta

    tdata = ilr(zdata)
    return tdata


def inv_ilr_transform_data(data):
    return ilr_inv(data)


def flatten_data(data):
    reads = data['reads']
    assign = data['assignments']
    # groups = ['jax', 'envigo', 'fmt'] # TODO: will need to change names, and also implement separately for time series...
    groups = ['pre_perturb', 'comparator', 'post_perturb']

    flatreads = None
    subj_labels = None #[]
    cluster_labels = None
    
    for ig, grp in enumerate(groups):
        subjs = list(reads[grp].keys())
        for isx, sub in enumerate(subjs):
            particles = reads[grp][sub]
            nparticles, notus = particles.shape
            clabs = assign[grp][sub]
            if flatreads is None:
                flatreads = particles
                cluster_labels = clabs
            else:
                flatreads = np.vstack([flatreads, particles])
                cluster_labels = np.concatenate([cluster_labels, clabs])
            for _ in range(nparticles):
                if subj_labels is None:
                    subj_labels = np.array([ig, isx])
                else:
                    subj_labels = np.vstack([subj_labels, np.array([ig, isx])])
                # gslabel = f"{grp}{sub}"
                # subj_labels.append(gslabel)
        
    return flatreads, subj_labels, cluster_labels
