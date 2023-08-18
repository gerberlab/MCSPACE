import numpy as np 
import pandas as pd 
import torch 
import os 
from pathlib import Path 
from mcspace.utils import pickle_load 


def get_basic_data(reads_in, device):
    bulk = reads_in.sum(axis=0)/reads_in.sum()
    contamination_communities = torch.from_numpy(bulk).to(dtype=torch.float, device=device)
    reads = torch.from_numpy(reads_in).to(torch.float)
    norm = torch.sum(reads, dim=1)
    rel_data = torch.div(reads, norm.unsqueeze(1))
    z_data = torch.log(rel_data+0.0001)
    z_std, z_mean = torch.std_mean(z_data, dim=1)
    z_data = z_data - z_mean.unsqueeze(1)
    z_data = torch.div(z_data, z_std.unsqueeze(1))
    if z_data.isnan().any():
        raise ValueError("nan in normed data")
    return {'count_data': reads.to(device), 'normed_data': z_data.to(device), 'full_normed_data': z_data.to(device)}, contamination_communities


def get_perturbation_data(reads, device):
    counts = {}
    contamination_communities = {}
    normed_data = {}
    # output L* x O; for all particles concatenated together
    full_normed_data = [] 
    for g in reads.keys():
        subjs = reads[g].keys()
        counts[g] = {}
        all_particles = None
        normed_data[g] = {}
        for s in subjs:
            subj_reads = reads[g][s]
            counts[g][s] = torch.from_numpy(subj_reads).to(dtype=torch.float, device=device)
            data, _ = get_basic_data(subj_reads, device)
            normed_data[g][s] = data['normed_data']
            full_normed_data.append(data['normed_data'])
            if all_particles is None:
                all_particles = reads[g][s]
            else:
                all_particles = np.concatenate([all_particles, reads[g][s]], axis=0)
        bulk = all_particles.sum(axis=0)/all_particles.sum()
        contamination_communities[g] = torch.from_numpy(bulk).to(dtype=torch.float, device=device)
    combined_data = torch.cat(full_normed_data, dim=0)
    return {'count_data': counts, 'normed_data': normed_data, 'full_normed_data': combined_data}, contamination_communities


def get_timeseries_data(reads, device):
    #! note, using one contamination community as average over time; can reconsider if given more subjects
    counts = {}
    normed_data = {}
    # output L* x O; for all particles concatenated together
    full_normed_data = [] 
    all_particles = None
    for t in reads.keys():
        counts[t] = torch.from_numpy(reads[t]).to(dtype=torch.float, device=device)
        data, _ = get_basic_data(reads[t], device)
        normed_data[t] = data['normed_data']
        full_normed_data.append(data['normed_data'])
        if all_particles is None:
            all_particles = reads[t]
        else:
            all_particles = np.concatenate([all_particles, reads[t]], axis=0)
    bulk = all_particles.sum(axis=0)/all_particles.sum()
    contamination_communities = torch.from_numpy(bulk).to(dtype=torch.float, device=device)
    combined_data = torch.cat(full_normed_data, dim=0)
    return {'count_data': counts, 'normed_data': normed_data, 'full_normed_data': combined_data}, contamination_communities
    

def get_data_for_inference(data, device):
    if type(data) == dict:
        keys = list(data.keys())
        if type(data[keys[0]]) == dict:
            model_data, contam_comms = get_perturbation_data(data, device)
        else:
            model_data, contam_comms = get_timeseries_data(data, device)
    else:   
        model_data, contam_comms = get_basic_data(data, device)
    return model_data, contam_comms


def estimate_group_variances(data):
    groups = data.keys()
    xvar = {}
    for grp in groups:
        grpdata = data[grp]
        subjects = list(grpdata.keys())
        nsubj = len(subjects)
        notus = grpdata[subjects[0]].shape[1]
        sdata = np.zeros((notus, nsubj))
        for i,s in enumerate(subjects):
            counts = grpdata[s]
            ra = counts.sum(axis=0)/counts.sum()
            sdata[:,i] = np.log(ra + 1e-20)
        if nsubj < 3:
            svarmed = 0.1
        else:
            svar = np.var(sdata, axis=1)
            svarmed = np.median(svar)
        xvar[grp] = svarmed
    return xvar 
