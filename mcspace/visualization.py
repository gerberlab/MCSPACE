import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib as mpl 
from matplotlib.lines import Line2D


#* global parameters
FONTSIZE = 20
LABEL_FONTSIZE=22
TICK_FONTSIZE = 16
ANNOT_SIZE = 15

LINEWIDTH = 5

BETA_CMAP = mpl.colormaps['bone'].reversed()
BETA_MIN = -2
BETA_MAX = 0

# pvalues
PVAL_CUTOFFS = [-1, 0.0001, 0.001, 0.01, 0.05, 1]
cmap = sns.color_palette("Blues", n_colors=5)
cmap.reverse()
ENRICH_CMAP = ListedColormap(cmap.as_hex())
ENRICH_NORM = BoundaryNorm(PVAL_CUTOFFS, ENRICH_CMAP.N)

# bayes factors
BF_LARGE = 110
BF_MED = 40
BF_SMALL = 10


#* methods
def _remove_border(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel('')
    ax.set_ylabel('')
    return ax


def add_enrichment_subplot(ax, pvals_all, counts_all, level):
    pvals = pvals_all[level]
    counts = counts_all[level].values
    
    sns.heatmap(pvals, cmap=ENRICH_CMAP, annot=counts, linewidth=0.1, linecolor='black', cbar=False, ax=ax, norm=ENRICH_NORM, yticklabels=False, annot_kws={'size': ANNOT_SIZE})
   
    ax.set_ylabel(level, fontsize=LABEL_FONTSIZE)
    ax.set_yticks(np.arange(pvals.shape[0])+0.5)
    names = list(pvals.index)
    names_trim = names #[name.replace('[Eubacterium] coprostanoligenes group', '[Eubacterium] coprostanoligenes\ngroup') for name in names]
    ax.set_yticklabels(names_trim, fontsize=FONTSIZE)
    ax.set_xticklabels([f"C{i+1}" for i in range(pvals.shape[1])], fontsize=FONTSIZE)
    ax.set_xlabel("Spatial community", fontsize=LABEL_FONTSIZE)
    ax.set_title("Community composition", fontsize=LABEL_FONTSIZE)

    return ax


def get_indicator_size(bf):
    if bf >= 100:
        return BF_LARGE
    if bf >= 10:
        return BF_MED
    if bf >= np.sqrt(10):
        return BF_SMALL
    return 0


def get_indicator_color(val):
    return 'red'


def add_ts_community_proportions_subplot(ax, beta, pert_mag, pert_bf):
    ax=sns.heatmap(np.log10(beta.T), ax=ax, cmap=BETA_CMAP, vmin=BETA_MIN, vmax=BETA_MAX, cbar=False, linewidth=LINEWIDTH,linecolor='black') #, square=True)
    ncomms, ntime = beta.shape
    for i in range(ncomms):
        for j in range(ntime-1):
            if np.abs(pert_bf[i,j]) >= np.sqrt(10):
                color = get_indicator_color(pert_mag[i,j])
                ind_size = get_indicator_size(pert_bf[i,j])
                ax.scatter(i + 0.5, j + 1 + 0.5, s=ind_size, c=color)    
    ax.set_xticklabels([f"C{i+1}" for i in range(ncomms)], fontsize=FONTSIZE)
    ax.set_yticklabels([f"T{i+1}" for i in range(ntime)], rotation=0, fontsize=FONTSIZE)
    ax.set_title("Community proportion", fontsize=LABEL_FONTSIZE)
    return ax


def add_pert_community_proportions_subplot(ax, beta, pert_mag, pert_bf):   
    ax=sns.heatmap(np.log10(beta.T), ax=ax, cmap=BETA_CMAP, vmin=BETA_MIN, vmax=BETA_MAX, cbar=False, linewidth=LINEWIDTH,linecolor='black') #, square=True)
    ncomms, ngrps = beta.shape
    for i in range(len(pert_bf)):
        if np.abs(pert_bf[i]) >= np.sqrt(10):
            color = get_indicator_color(pert_mag[i])
            ind_size = get_indicator_size(pert_bf[i])
            ax.scatter(i + 0.5, 2 + 0.5, s=ind_size, c=color)    
    ax.set_xticklabels([f"C{i+1}" for i in range(ncomms)], fontsize=FONTSIZE)
    ax.set_yticks([0.5,1.5,2.5])
    ax.set_yticklabels(["Preperturb", "Comparator", "Post perturb"], rotation=0, fontsize=FONTSIZE)
    ax.set_title("Community proportion", fontsize=LABEL_FONTSIZE)
    return ax


def make_legend(pval_ax, cbar_ax, bf_ax):
    # pvalues    
    pval_ax.set_title("Adjusted\np-values", fontsize=FONTSIZE)  
    cbar_pval = mpl.colorbar.ColorbarBase(pval_ax, cmap=ENRICH_CMAP, norm=ENRICH_NORM, orientation='vertical')
    x = np.array([-1, 0.0001, 0.001, 0.01, 0.05, 1])
    t = 0.5*(x[1:] + x[:-1])
    cbar_pval.ax.set_yticks(t)
    cbar_pval.ax.set_yticklabels(['p<0.0001', 'p<0.001', 'p<0.01', 'p<0.05', 'ns'])
    cbar_pval.ax.tick_params(labelsize=TICK_FONTSIZE)
    
    # beta - abundance
    cbar_ax.set_title("Log\nabundance", fontsize=FONTSIZE)
    # TODO: add arrows for over/under-flow
    beta_norm = mpl.colors.Normalize(vmin=BETA_MIN, vmax=BETA_MAX)
    cbar_beta = mpl.colorbar.ColorbarBase(cbar_ax, cmap=BETA_CMAP, norm=beta_norm, orientation='vertical')
    cbar_beta.ax.tick_params(labelsize=TICK_FONTSIZE)
    
    # bayes factors
    zrs = np.ones((3,1))*np.nan
    bf_ax = sns.heatmap(zrs, linewidth=0.1, linecolor='black', cbar=False, ax=bf_ax)
    ind_sizes = [110, 40, 10]
    for i in range(3):
        bf_ax.scatter(0.5, i + 0.5, s=ind_sizes[i], c='red')
    bf_ax.set_xticklabels('')
    bf_ax.set_xticks([])
    bf_ax.set_yticklabels(['BF > 100', '10 < BF < 100', 'sqrt(10) < BF < 10'], rotation=0)
    bf_ax.yaxis.set_label_position("right")
    bf_ax.yaxis.tick_right()
    bf_ax.set_title("Bayes\nfactors", fontsize=FONTSIZE)

    return pval_ax, cbar_ax, bf_ax


def make_enrichment_summary_figure(pvals, counts, beta, pert_mag, pert_bf, level, case):
    # make main figure
    fig = plt.figure(figsize=(20,20))
    gs = fig.add_gridspec(nrows=3,ncols=3,
                        width_ratios=(1,0.3,1),
                        height_ratios=(1,1,2),
                        wspace=0.05,
                        hspace=0.05)

    # add subplots
    ax_fam = fig.add_subplot(gs[:,0])
    ax_beta = fig.add_subplot(gs[0,2])

    # add enrichment plot
    ax_fam = add_enrichment_subplot(ax_fam, pvals, counts, level=level)

    # add community proportions and perturbation indicators
    if case == "perturbation":
        ax_beta = add_pert_community_proportions_subplot(ax_beta, beta, pert_mag, pert_bf)
    if case == "time_series":
        ax_beta = add_ts_community_proportions_subplot(ax_beta, beta, pert_mag, pert_bf)

    # subgridspec for legend
    gs_lgd = gs[2,2]

    # add legend subplots
    ax_lgd = fig.add_subplot(gs[2,2])
    ax_lgd = _remove_border(ax_lgd)
    ax_lgd.set_title("Legend", fontsize=LABEL_FONTSIZE) #,  y=1.0, pad=-14)
    
    gscbar = gs_lgd.subgridspec(3,7)
    lgd_ax = fig.add_subplot(gscbar[0,:], facecolor=None)
    lgd_ax = _remove_border(lgd_ax)
    pval_ax = fig.add_subplot(gscbar[1,0])
    cbar_ax = fig.add_subplot(gscbar[1,3])
    bf_ax = fig.add_subplot(gscbar[1,5])

    pval_ax, cbar_ax, bf_ax = make_legend(pval_ax, cbar_ax, bf_ax)
    return fig


def make_ts_enrichment_summary_figure(pvals, counts, beta, pert_mag, pert_bf, level):
    fig = make_enrichment_summary_figure(pvals, counts, beta, pert_mag, pert_bf, level, "time_series")
    return fig


def make_pert_enrichment_summary_figure(pvals, counts, beta, pert_mag, pert_bf, level):
    fig = make_enrichment_summary_figure(pvals, counts, beta, pert_mag, pert_bf, level, "perturbation")
    return fig



#* TREE PLOTS ============================================================================================


