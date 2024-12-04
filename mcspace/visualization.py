from mcspace.utils import pickle_load, get_subj_averaged_assemblage_proportions, \
    get_lowest_level_name, get_assoc_scores, filter_assoc_scores, update_names, \
    output_association_network_to_graphML
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import mcspace.vis_tools as vt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path
import ete3
from Bio import Phylo
mpl.use('agg')
mpl.rcParams['font.sans-serif'] = "Arial"
mpl.rcParams['font.family'] = "sans-serif"
plt.rcParams['svg.fonttype'] = 'none'


def render_assemblages(results,
                    outfile,
                    otu_threshold=0.05,
                    treefile=None,
                    fontsize=6,
                    legend=True):

    def _get_pruned_tree(treefile, taxaids, temppath=Path("./_tmp")):
        tree = ete3.Tree((treefile).as_posix())
        print("original tree size:", len(tree))
        tree.prune(taxaids, True)
        print("pruned tree size:", len(tree))

        treeout = "tree.nhx"
        temppath.mkdir(exist_ok=True, parents=True)
        tree.write(outfile=(temppath / treeout).as_posix())
        phylo_tree = Phylo.read(temppath / treeout, format='newick')
        return phylo_tree

    def _update_names(otus, full_taxonomy):
        prefix_taxa = {'Species': '*', 'Genus': '**', 'Family': '***', 'Order': '****',
                       'Class': '*****', 'Phylum': '******', 'Domain': '*******'}
        new_names = []
        for text in otus:
            otu_name = str(text._text).replace(' ', '').capitalize()
            name, level = vt.get_lowest_level(otu_name, full_taxonomy)
            prefix = prefix_taxa[level]
            label = prefix + name + ' ' + otu_name.upper()
            new_names.append(label)
        return new_names

    theta_vmin = np.floor(np.log10(otu_threshold))
    theta_cmap = mcolors.LinearSegmentedColormap.from_list('theta cmap',
                                                           [(0, 'lightyellow'),
                                                            (0.7, 'yellowgreen'),
                                                            (1, 'green')], N=256)
    theta_cmap.set_under('white')

    #! will need to make sure these are in taxonomy...
    taxlevels = ['Otu', 'Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    theta = results['assemblages'].reset_index()
    taxonomy = theta[taxlevels].copy()
    taxonomy = taxonomy.set_index("Otu")
    thetadf = theta.set_index(taxlevels)

    # apply OTU threshold
    otu_sub = thetadf.index[(thetadf > otu_threshold).any(axis=1)].get_level_values("Otu")
    otu_order = otu_sub # unless plotting tree; in which case update
    # zero out below threshold
    thetadf2 = thetadf.copy()
    thetadf2[thetadf2<otu_threshold] = 0
    beta_order=thetadf.columns

    # create figure object
    # scale by number of cells include tree
    notus = len(otu_order)
    ncomm = len(beta_order)
    aspect_ratio = notus/ncomm
    tree_ratio = 1.0 #1.0/1.3
    fig = plt.figure(figsize=(8.5,8.5*aspect_ratio/tree_ratio))

    gs = fig.add_gridspec(nrows=2, ncols=3,
                          width_ratios=[0.1,0.2,1],
                          height_ratios=[1,0.03],
                          hspace=0.3)

    if treefile is not None:
        ax_tree = fig.add_subplot(gs[0,0])
        tree = _get_pruned_tree(treefile, otu_order, temppath=Path("./_tmp_sub"))
        ax_tree, otu_order = vt.plot_phylo_tree(ax_tree, tree, taxonomy, fontsize=fontsize)
    else:
        ax_tree = None

    ax_theta = fig.add_subplot(gs[0,2])

    if legend is True:
        legend_gs = gs[1,2].subgridspec(1,4, width_ratios=[2,1,1,1])
        ax_cbar = fig.add_subplot(legend_gs[0,3])
        ax_taxa_lgd = fig.add_subplot(legend_gs[0,1])
    else:
        ax_cbar = None

    ax_theta = vt.plot_assemblages(ax_theta, thetadf2,
                                   otu_order, beta_order,
                                   cmap=theta_cmap, vmin=theta_vmin, vmax=0,
                                   cbar=False, yticklabels=True)
    ax_theta.set_xticklabels(ax_theta.get_xticklabels(), fontsize=fontsize, rotation=90)
    ax_theta.set_xlabel("Assemblage", fontsize=fontsize)

    if treefile is not None:
        ax_theta.set_yticks([])
        ax_theta.set_ylabel("")
    else:
        new_labels = _update_names(ax_theta.get_yticklabels(), taxonomy)
        ax_theta.set_yticklabels(new_labels, fontsize=fontsize)
        ax_theta.set_ylabel("OTU", fontsize=fontsize)
    ax_theta = vt.add_border(ax_theta)

    if legend is True:
        norm = mpl.colors.Normalize(vmin=theta_vmin, vmax=0)
        ax_cbar= mpl.colorbar.ColorbarBase(ax_cbar, cmap=theta_cmap, norm=norm, orientation='horizontal')
        ax_cbar.ax.set_title("Log10 probability in assemblage", fontsize=fontsize)
        xlabelvalues = np.arange(theta_vmin,0.1,1).astype(int)
        ax_cbar.ax.set_xticks(xlabelvalues)
        ax_cbar.ax.set_xticklabels(xlabelvalues, fontsize=fontsize)

        lgd_xpos = 0.5
        lgd_ypos = 2.5
        indent_xpos = lgd_xpos + 0.1
        dy = 0.5
        lgd_fontsize = fontsize

        levels = ['Species', 'Genus', 'Family', 'Order', 'Class', 'Phylum', 'Domain']
        level_dict = {'Species': '*', 'Genus': '**', 'Family': '***',
                      'Order': '****', 'Class': '*****', 'Phylum': '******', 'Domain': '*******'}

        ax_taxa_lgd.text(lgd_xpos, lgd_ypos, "Taxonomy keys", fontsize=lgd_fontsize)
        for i, level in enumerate(levels):
            ax_taxa_lgd.text(indent_xpos, lgd_ypos - (i + 1) * dy, f"{level_dict[level]}: {level}",
                             fontsize=lgd_fontsize)
        ax_taxa_lgd = vt.remove_border(ax_taxa_lgd)
    plt.savefig(outfile)

    # return axis objects for tree and theta
    return ax_tree, ax_theta, ax_cbar

# render assemblage proportions ==============================
def _get_indicator_size(bf):
    # bayes factors
    BF_LARGE = 10
    BF_MED = 4
    BF_SMALL = 1

    if bf >= 100:
        return BF_LARGE
    if bf >= 10:
        return BF_MED
    if bf >= np.sqrt(10):
        return BF_SMALL
    return 0

def add_subj_bf_annotation(ax, betadf, t, bayes_factors,beta_order):
    nsubj = len(betadf['Subject'].unique())
    bfs = bayes_factors.loc[beta_order,t].values
    ncomm = len(bfs)
    for i in range(nsubj):
        for j in range(ncomm):
            ind_size = _get_indicator_size(bfs[j])
            ax.scatter(j+0.5,i+0.5,s=ind_size,c='goldenrod')
    return ax


def add_all_bf_annotation(ax, average_beta, bayes_factors, border):
    times = average_beta.index
    ncomm = len(average_beta.columns)
    ptimes = bayes_factors.columns
    for i,t in enumerate(times):
        if t in ptimes:
            bfs = bayes_factors.loc[border, t].values
            for j in range(ncomm):
                ind_size = _get_indicator_size(bfs[j])
                ax.scatter(j+0.5,i+0.5,s=ind_size,c='goldenrod')
    return ax


def render_assemblage_proportions(results,
                                  outfile,
                                  average_subjects=False,
                                  annotate_bayes_factors=False,
                                  logscale=True,
                                  beta_vmin=-3,
                                  fontsize=6,
                                  legend=True):
    beta = results['assemblage_proportions']
    bayes_factors = results['perturbation_bayes_factors']
    pert_times = bayes_factors.columns
    times = beta['Time'].unique()
    subjects = beta['Subject'].unique()
    ntime = len(times)
    nsubj = len(subjects)
    if average_subjects is False:
        nplots = ntime
    else:
        nplots = 1
        nsubj = 1


    beta_cmap = 'Blues'
    if logscale is True:
        beta_vmax=0
    else:
        beta_vmax=1

    taxlevels = ['Otu', 'Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    theta = results['assemblages'].reset_index()
    thetadf = theta.set_index(taxlevels)
    beta_order = thetadf.columns

    if average_subjects is False:
        hratios = [nsubj]*ntime
    else:
        hratios = [ntime]
    hratios.append(0.3*ntime*4)
    hratios.append(0.1*ntime*2)
    ncomm = len(beta_order)
    aspect_ratio = (ntime*nsubj)/ncomm
    plot_ratio = (ntime*nsubj + ntime)/(ntime*nsubj)
    fig = plt.figure(figsize=(8.5,8.5*aspect_ratio*plot_ratio))
    gs = fig.add_gridspec(nrows=nplots + 2, ncols=1,
                          height_ratios=hratios,
                          hspace=0.2)

    if legend is True:
        legend_gs = gs[-1, 0].subgridspec(1, 4, width_ratios=[2,1,0.5,1])
        if annotate_bayes_factors is True:
            ax_bf = fig.add_subplot(legend_gs[0,1])
        else:
            ax_bf = None
        ax_cbar = fig.add_subplot(legend_gs[0,3])
    else:
        ax_cbar = None
        ax_bf = None

    ax = []
    for i in range(nplots):
        ax.append(fig.add_subplot(gs[i, 0]))

    if average_subjects is False:
        for i,t in enumerate(times):
            if i == (ntime-1):
                ax[i]=vt.plot_all_subject_proportions_at_timepoint(ax[i],beta,t,order=beta_order,
                                                                     logscale=logscale,vmin=beta_vmin,
                                                                   vmax=beta_vmax,
                                                                   xticklabels=True)
                ax[i].set_xticklabels(ax[i].get_xticklabels(),fontsize=fontsize,rotation=90)
                ax[i].set_xlabel("Assemblage", fontsize=fontsize)
            else:
                ax[i]=vt.plot_all_subject_proportions_at_timepoint(ax[i],beta,t,order=beta_order,
                                                                     logscale=logscale,vmin=beta_vmin,vmax=beta_vmax,)

            # annotate bayes factors
            if (annotate_bayes_factors is True) and (bayes_factors is not None):
                if t in pert_times:
                    ax[i]=add_subj_bf_annotation(ax[i],beta,t,bayes_factors,beta_order)
            ax[i].set_ylabel("")
            ax[i].set_yticklabels(ax[i].get_yticklabels(), fontsize=fontsize)
            ax[i]=vt.add_border(ax[i])
            ax[i].set_ylabel(f"Time: {t}", rotation=0, labelpad=50, ha='left', fontsize=fontsize)
    else:
        # get subject averaged beta
        avebeta = get_subj_averaged_assemblage_proportions(beta)
        if logscale is True:
            plot_vals = np.log10(avebeta.loc[times,beta_order])
        else:
            plot_vals = avebeta.loc[times,beta_order]
        # plot heatmap
        ax[0]=sns.heatmap(plot_vals, ax=ax[0], cmap=beta_cmap, vmin=beta_vmin, vmax=beta_vmax,
                          linecolor ='#e6e6e6', linewidth=0.5,xticklabels=True,yticklabels=True,
                          square=False,cbar=False)
        ax[0].set_xticklabels(ax[0].get_xticklabels(), fontsize=fontsize, rotation=90)
        ax[0].set_yticklabels(ax[0].get_yticklabels(), fontsize=fontsize, rotation=0)
        ax[0].set_xlabel("Assemblage", fontsize=fontsize)
        ax[0].set_ylabel("Time", fontsize=fontsize)
        ax[0]=vt.add_border(ax[0])
        # annotate bayes factors
        if (annotate_bayes_factors is True) and (bayes_factors is not None):
            ax[0]=add_all_bf_annotation(ax[0],avebeta,bayes_factors,beta_order)

    # LEGEND
    if legend is True:
        norm = mpl.colors.Normalize(vmin=beta_vmin, vmax=beta_vmax)
        ax_cbar= mpl.colorbar.ColorbarBase(ax_cbar, cmap=beta_cmap, norm=norm, orientation='horizontal')
        if logscale:
            ax_cbar.ax.set_title("Log10 assemblage proportion", fontsize=fontsize)
        else:
            ax_cbar.ax.set_title("Assemblage proportion", fontsize=fontsize)
        ax_cbar.ax.set_xticklabels(ax_cbar.ax.get_xticklabels(), fontsize=fontsize)
        if annotate_bayes_factors is True:
            zrs = np.ones((1,3)) * np.nan
            ax_bf = sns.heatmap(zrs, linewidth=0.1, linecolor='black', cbar=False, ax=ax_bf)
            ind_sizes = [1, 4, 10]
            for i in range(3):
                ax_bf.scatter(i + 0.5, 0.5, s=ind_sizes[i], c='goldenrod')
            ax_bf.set_xticklabels([r'$\sqrt{10} \leq BF < 10$',r'$10 \leq BF < 100$',r'$BF > 100$'],fontsize=fontsize,rotation=20,
                                  ha='right',rotation_mode='anchor')
            ax_bf.set_title("Bayes factor",fontsize=fontsize)
    plt.savefig(outfile,bbox_inches="tight")

    return ax, ax_cbar, ax_bf


# export association networks to cytoscape =======================
def export_association_networks_to_cytoscape(oidx,
                                             results,
                                             outfile,
                                             ra_threshold=0.01,
                                             edge_threshold=0.01):
    """

    Parameters
    ----------
    oidx: otu index of hub taxon for which associations are computed and exported
    results: pickle file of mcspace inference results
    outfile: name of xml file to output association networks to
    ra_threshold: relative abundance threshold for which taxa to keep (taxa with relative abundance
        below the given threshold on all time points are removed)
    edge_threshold: threshold for which edges to keep (taxa with an edge less than the threshold on
        all time points are removed)

    Returns
    -------
    xml file giving subject averaged associations of key taxon over time
    """
    beta = results['assemblage_proportions']
    times = beta['Time'].unique()
    subjects = beta['Subject'].unique()
    taxlevels = ['Otu', 'Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    theta = results['assemblages'].reset_index()
    taxonomy = theta[taxlevels].copy()
    taxonomy = taxonomy.set_index("Otu")
    thetadf = theta.set_index(taxlevels)
    beta_order = thetadf.columns
    avebeta = get_subj_averaged_assemblage_proportions(beta)
    radf = results['relative_abundances']
    radf.columns = radf.columns.astype(int)

    otu_name = get_lowest_level_name(oidx, taxonomy)

    # get edges and node weights
    alpha = get_assoc_scores(thetadf, avebeta, oidx)

    ew = filter_assoc_scores(alpha, radf, oidx,
                                ra_threshold=ra_threshold, edge_threshold=edge_threshold)
    nw = radf.loc[ew.index, :]

    # update labels for taxa
    nw3 = update_names(nw, taxonomy)
    ew3 = update_names(ew, taxonomy)
    # output to file
    output_association_network_to_graphML(oidx, nw3, ew3, taxonomy, outfile)
