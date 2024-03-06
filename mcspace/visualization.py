import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np


def get_clustered_otu_assemblage_ordering(theta):
    # TODO: probably want to change this to something else...
    cg = sns.clustermap(np.log10(theta.T), cmap='Blues', vmin=-2, vmax=0)
    plt.close()
    otu_order = cg.dendrogram_row.reordered_ind
    assemblage_order = cg.dendrogram_col.reordered_ind
    return otu_order, assemblage_order


def _get_indicator_size(bf):
    # bayes factors
    BF_LARGE = 110
    BF_MED = 40
    BF_SMALL = 10

    if bf >= 100:
        return BF_LARGE
    if bf >= 10:
        return BF_MED
    if bf >= np.sqrt(10):
        return BF_SMALL
    return 0


def annotate_perturbation_bayes_factors(ax, assemblage_order, perturbation_bayes_factors):    
    n_assemblage, n_pert = perturbation_bayes_factors.shape
    pert_bf_reordered = perturbation_bayes_factors[assemblage_order,:]
    for i in range(n_assemblage):
        for j in range(n_pert):
            if np.abs(pert_bf_reordered[i,j]) >= np.sqrt(10):
                color = 'darkgoldenrod' #get_indicator_color(pert_mag[i,j])
                ind_size = _get_indicator_size(pert_bf_reordered[i,j])
                ax.scatter(i + 0.5, j + 0.5, s=ind_size, c=color)   
    return ax


def update_otu_labels(ax, taxonomy):
    def _get_lowest_level(otu, taxonomy):
        taxonomies = ['Genus', 'Family', 'Order', 'Class', 'Phylum', 'Kingdom']
        for level in taxonomies:
            levelid = taxonomy.loc[otu,level]
            if levelid != 'na':
                return levelid, level

    prefix_taxa = {'Genus': '*', 'Family': '**', 'Order': '***', 'Class': '****'
                , 'Phylum': '*****', 'Kingdom': '******'}

    ylabels = []
    for text in ax.get_yticklabels():
        taxonname = str(text._text).replace(' ','')
        otu_name = taxonname
        name, level = _get_lowest_level(taxonname, taxonomy)
        prefix = prefix_taxa[level]
        taxonname = ' ' + prefix + ' ' + name + ' ' + otu_name
        ylabels.append(taxonname)

    ax.set_yticklabels(ylabels, rotation=0)
    return ax


def plot_assemblages(ax, theta, taxonomy, otu_order, assemblage_order, cmap=mpl.colormaps['Blues'], vmin=-2, vmax=0,
                    linewidth=0.5, linecolor='#e6e6e6', logscale=True, cbar=True, square=True):
    n_assemblages, n_otus = theta.shape
    assemblages = [f'A{i+1}' for i in range(n_assemblages)]

    to_plot = (theta[assemblage_order,:][:,otu_order]).T
    if logscale:
        to_plot = np.log10(to_plot)

    og_otuinds = taxonomy.index
    otuinds = og_otuinds[otu_order]

    df = pd.DataFrame(data=to_plot, index=otuinds, columns=assemblages)
    ax = sns.heatmap(df, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, cbar=cbar,
                    linewidth=linewidth, linecolor=linecolor, xticklabels=True, yticklabels=True, square=square)

    # # Drawing the frame
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(linewidth)
        spine.set_color('black')
    return ax


def plot_assemblage_proportions(ax, beta, assemblage_order, subject, cmap=mpl.colormaps['Blues'], vmin=-2, vmax=0,
                                linecolor='#e6e6e6', logscale=True, cbar=True, square=True):
    to_plot = beta[assemblage_order,:,subject].T
    if logscale is True:
        to_plot = np.log10(to_plot)

    ax = sns.heatmap(to_plot, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, cbar=cbar, linewidth=1.0, 
                     linecolor=linecolor, square=square, xticklabels=True, yticklabels=True)
    
    # frame
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color('black')
            
    return ax


#* Legend components
def plot_colorbar(ax, cmap, vmin, vmax):
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    ax=mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')
    return ax


def plot_bayes_factors_key(ax, linewidth=0.1, linecolor='black', markercolor='darkgoldenrod'):
    zrs = np.ones((3,1))*np.nan
    ax = sns.heatmap(zrs, linewidth=linewidth, linecolor=linecolor, cbar=False, ax=ax)
    ind_sizes = [110, 40, 10]
    for i in range(3):
        ax.scatter(0.5, i + 0.5, s=ind_sizes[i], c=markercolor)
    return ax


def render_perturbation_effect_and_assemblages(pert_bf, betadiff, theta, taxonomy, otu_order, assemblage_order):
    scale = 1
    fig = plt.figure(figsize=(8.5*scale,11*scale*1.2))
    gs = fig.add_gridspec(nrows=3,ncols=2,
                        width_ratios=(1.0,0.3),
                        height_ratios=(0.5,0.5,20), #! figure out automatic scaling...
                        wspace=0.05,
                        hspace=0.05)

    ax_bf = fig.add_subplot(gs[0,0])
    ax_beta = fig.add_subplot(gs[1,0])
    ax = fig.add_subplot(gs[2,0])

    theta_cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["white","green"])
    theta_vmin = -2
    theta_vmax = 0
    theta_linewidth = 0.5
    linecolor = '#e6e6e6'
    theta_logscale=True
    cbar=False #True
    square=False

    ncomm = pert_bf.shape[0]
    ax_bf=sns.heatmap(np.nan*np.ones((ncomm,1)).T, linewidth=1,linecolor='k',cbar=False, ax=ax_bf)
    ax_bf=annotate_perturbation_bayes_factors(ax_bf,assemblage_order,pert_bf)
    ax_bf.set_xticks([])
    ax_bf.set_yticklabels(["Perturbation BF"], rotation=0)

    beta_cmap = 'seismic_r'
    ax_beta=plot_assemblage_proportions(ax_beta, betadiff[:,None,:], assemblage_order, subject=0, cmap=beta_cmap, vmin=-1,
                                    vmax=1, logscale=False, cbar=False, square=False)
    ax_beta.set_xticks([])
    ax_beta.set_yticklabels(["Change in proportion"], rotation=0)
    ax=plot_assemblages(ax,theta, taxonomy, otu_order, assemblage_order, cmap=theta_cmap, vmin=theta_vmin, vmax=theta_vmax,
                    linewidth=theta_linewidth, linecolor=linecolor, logscale=theta_logscale, cbar=cbar, square=square)
    ax=update_otu_labels(ax,taxonomy)
    ax.set_ylabel("")
    ax.set_xlabel("Spatial assemblage")

    # legend
    gs_lgd = gs[:,1].subgridspec(5,3, hspace=0.5)
    bf_lgd = fig.add_subplot(gs_lgd[0,1])
    beta_lgd = fig.add_subplot(gs_lgd[1,1])
    theta_lgd = fig.add_subplot(gs_lgd[2,1])
    # tax_lgd = fig.add_subplot(gs_lgd[3,1])

    # bf legend
    bf_lgd=plot_bayes_factors_key(bf_lgd)
    bf_lgd.set_xticks([])
    bf_lgd.set_yticks([0.5,1.5,2.5])
    bf_lgd.set_yticklabels([r'BF $> 100$', r'$10 <$ BF $< 100$', r'$\sqrt{10} <$ BF $< 10$'], rotation=0)
    bf_lgd.yaxis.tick_right()
    bf_lgd.set_title("Perturbation\nBayes Factor")

    # beta legend
    beta_lgd=plot_colorbar(beta_lgd,beta_cmap,-1,1)
    beta_lgd.ax.set_title("Change in\nproportion")

    # theta legend
    theta_lgd=plot_colorbar(theta_lgd,theta_cmap,theta_vmin,theta_vmax)
    theta_lgd.ax.set_title("Relative\nabundance")
    theta_lgd.ax.set_yticks([0, -1, -2])
    theta_lgd.ax.set_yticklabels([r'$10^{0}$',r'$10^{-1}$',r'$10^{-2}$'])

    return fig, ax_bf, ax_beta, ax, bf_lgd, beta_lgd, theta_lgd


def render_proportions_and_assemblages(beta, theta, taxonomy, otu_order, assemblage_order, ylabels=""):
    # TODO: also want to add legend...
    scale = 1
    ncomm, ntime, nsubj = beta.shape
    fig = plt.figure(figsize=(8.5*scale,11*scale*1.2))
    gs = fig.add_gridspec(nrows=2,ncols=2,
                        width_ratios=(1.0,0.3),
                        height_ratios=(ntime,20), #! figure out automatic scaling...
                        wspace=0.05,
                        hspace=0.05)

    ax_beta = fig.add_subplot(gs[0,0])
    ax = fig.add_subplot(gs[1,0])

    theta_cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["white","green"])
    theta_vmin = -2
    theta_vmax = 0
    theta_linewidth = 0.5
    linecolor = '#e6e6e6'
    theta_logscale=True
    cbar=False #True
    square=False

    beta_max = np.amax(beta)
    beta_cmap = "gray_r"
    ax_beta=plot_assemblage_proportions(ax_beta, beta, assemblage_order, subject=0, cmap=beta_cmap, vmin=0,
                                    vmax=beta_max, logscale=False, cbar=False, square=False)
    ax_beta.set_xticks([])
    ax_beta.set_yticklabels(ylabels, rotation=0)

    ax=plot_assemblages(ax,theta, taxonomy, otu_order, assemblage_order, cmap=theta_cmap, vmin=theta_vmin, vmax=theta_vmax,
                    linewidth=theta_linewidth, linecolor=linecolor, logscale=theta_logscale, cbar=cbar, square=square)
    ax=update_otu_labels(ax,taxonomy)
    ax.set_ylabel("")
    ax.set_xlabel("Spatial assemblage")
    
    # legend
    gs_lgd = gs[:,1].subgridspec(5,3, hspace=0.5)
    # bf_lgd = fig.add_subplot(gs_lgd[0,1])
    beta_lgd = fig.add_subplot(gs_lgd[0,1])
    theta_lgd = fig.add_subplot(gs_lgd[1,1])
    # tax_lgd = fig.add_subplot(gs_lgd[3,1])

    # beta legend
    beta_lgd=plot_colorbar(beta_lgd,beta_cmap,0,beta_max) 
    beta_lgd.ax.set_title("Assemblage\nproportion")

    # theta legend
    theta_lgd=plot_colorbar(theta_lgd,theta_cmap,theta_vmin,theta_vmax)
    theta_lgd.ax.set_title("Relative\nabundance")
    theta_lgd.ax.set_yticks([0, -1, -2])
    theta_lgd.ax.set_yticklabels([r'$10^{0}$',r'$10^{-1}$',r'$10^{-2}$'])
    return fig, ax_beta, ax, beta_lgd, theta_lgd
