import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from statsmodels.sandbox.stats.multicomp import multipletests
import scipy.stats as stats
from sklearn.metrics import auc
import torch
from pathlib import Path
from mcspace.utils import pickle_load, pickle_save
import time


def get_gt_assoc(theta, otu_threshold):
    K, notus = theta.shape
    gt_assoc = 0
    for kidx in range(K):
        gt_assoc += np.outer(theta[kidx,:] > otu_threshold, theta[kidx,:] > otu_threshold)
    for oidx in range(notus):
        # remove self-assoc
        gt_assoc[oidx,oidx] = 0 
    
    gt_assoc[gt_assoc>0.5] = 1 #* 1 or 0; don't need more than 1

    return gt_assoc


def calc_auc(gt_assoc, post_probs, nthres = 100):
    notus = gt_assoc.shape[0]
    # take upper triangular matrices
    gta = (gt_assoc[np.triu_indices(notus, k=1)] > 0.5)
    pp = post_probs[np.triu_indices(notus, k=1)]
    thresholds = np.linspace(-0.001, 1.001, nthres)

    true_pos = np.zeros((nthres,))
    false_pos = np.zeros((nthres,))
    true_neg = np.zeros((nthres,))
    false_neg = np.zeros((nthres,))
    
    for i,thres in enumerate(thresholds):
        lta = (pp > thres)
        tp = (gta & lta).sum()
        fp = ((~gta) & lta).sum()
        tn = ((~gta) & (~lta)).sum()
        fn = ((gta) & (~lta)).sum()

        true_pos[i] = tp
        false_pos[i] = fp
        true_neg[i] = tn
        false_neg[i] = fn
    
    tpr = true_pos/(true_pos + false_neg)
    fpr = false_pos/(false_pos + true_neg)

    auc_val = auc(fpr, tpr)
    return auc_val, true_pos, false_pos, true_neg, false_neg
