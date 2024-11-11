import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import ete3
from Bio import SeqIO, Phylo
import networkx as nx


def remove_border(ax):
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


def add_border(ax, color='black'):
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color(color)
    return ax


def get_abundance_order(betadf):
    betadf_drop = betadf[['Assemblage', 'Value']]
    aveval = betadf_drop.groupby('Assemblage').mean()
    beta_order = aveval.sort_values(by='Value', ascending=False).index
    return beta_order


def plot_all_subject_proportions_at_timepoint(ax, betadf, time, order, logscale=True, vmin=-2, vmax=0, cmap='Blues',
                                   linecolor ='#e6e6e6', linewidth=0.5, xticklabels=False, yticklabels=True,
                                   square=False, cbar=False):
    if logscale is True:
        val = 'log10Value'
    else:
        val = 'Value'
        
    betasub = betadf.loc[(betadf['Time'] == time),:]
    betamatrix = betasub.pivot(index='Subject', columns='Assemblage', values=val)
    ax=sns.heatmap(betamatrix.loc[:,order], ax=ax, cmap=cmap, square=square, vmin=vmin, vmax=vmax, cbar=cbar,
               linewidth=linewidth, linecolor=linecolor, xticklabels=xticklabels, yticklabels=yticklabels)
    return ax


def plot_subject_proportions_timeseries(ax, betadf, subj, order, logscale=True, vmin=-2, vmax=0, cmap='Blues',
                                   linecolor ='#e6e6e6', linewidth=0.5, xticklabels=False, yticklabels=True,
                                   square=False, cbar=False):
    if logscale is True:
        val = 'log10Value'
    else:
        val = 'Value'
        
    betasub = betadf.loc[(betadf['Subject'] == subj),:]
    betamatrix = betasub.pivot(index='Time', columns='Assemblage', values=val)
    ax=sns.heatmap(betamatrix.loc[:,order], ax=ax, cmap=cmap, square=square, vmin=vmin, vmax=vmax, cbar=cbar,
               linewidth=linewidth, linecolor=linecolor, xticklabels=xticklabels, yticklabels=yticklabels)
    return ax


def get_pruned_tree(treepath, treefile, taxonomy, temppath=Path("./_tmp"), upper=False):
    tree = ete3.Tree((treepath / treefile).as_posix())
    print("original tree size:", len(tree))
    taxaids = list(taxonomy.index)
    if upper:
        taxaids = [idx.upper() for idx in taxaids]
    tree.prune(taxaids, True)
    print("pruned tree size:", len(tree))
    
    treeout = "tree.nhx"
    temppath.mkdir(exist_ok=True, parents=True)
    tree.write(outfile=(temppath / treeout).as_posix())
    phylo_tree = Phylo.read(temppath / treeout, format='newick')
    return phylo_tree


def get_lowest_level(otu, taxonomy):
    taxonomies = ['Species','Genus', 'Family', 'Order', 'Class', 'Phylum', 'Domain']
    for level in taxonomies:
        levelid = taxonomy.loc[otu,level]
        if levelid != 'na':
            return levelid, level
    

def plot_phylo_tree(ax, tree, taxonomy, fontsize=16, text_len=41):            
    TEXT_LEN=text_len
    prefix_taxa = {'Species': '*', 'Genus': '**', 'Family': '***', 'Order': '****',
                   'Class': '*****', 'Phylum': '******', 'Domain': '*******'}

    Phylo.draw(tree, axes=ax, do_show=False, show_confidence=False)
    taxa_order = []

    for text in ax.texts:
        taxonname = str(text._text).replace(' ','').capitalize()
        otu_name = taxonname
        taxa_order.append(otu_name)
        name, level = get_lowest_level(otu_name, taxonomy)
        prefix = prefix_taxa[level]
        taxonname = ' ' + prefix + ' ' + name + ' ' + otu_name.upper()
        text._text = taxonname
        text._text = text._text + '- ' * (TEXT_LEN - len(text._text))
        text.set_fontsize(fontsize)
        if (level == 'Species') or (level == 'Genus'):
            text.set_fontstyle('italic')
    ax = remove_border(ax)
    return ax, taxa_order


def plot_assemblages(ax, thetadf, otu_order, beta_order, cmap=None, logscale=True, vmin=-2, vmax=0,
               linecolor ='#e6e6e6', linewidth=0.5, xticklabels=True, yticklabels=False,
               square=False, cbar=False):
    
    # reset multiindex
    thetatemp = thetadf.reset_index()
    thetatemp=thetatemp.set_index('Otu')
    
    if logscale is True:
        thetaplot = np.log10(thetatemp.loc[otu_order,:][beta_order] + 1e-20)
    else:
        thetaplot = thetatemp.loc[otu_order,:][beta_order]
    
    if cmap is None:
        green = sns.light_palette("green", reverse=False, as_cmap=True)
        green.set_under('white')
        cmap=green
    cbar_kws = dict(extend='min')

    ax=sns.heatmap(thetaplot, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, linecolor=linecolor, linewidth=linewidth,
               xticklabels=xticklabels, yticklabels=yticklabels, square=square, cbar=cbar, cbar_kws=cbar_kws)
    return ax



# #! plotting association changes
# def get_assemblages_containing_taxon(thetadf, oidx, otu_threshold=0.05):
#     thetasub = thetadf.loc[thetadf.index.get_level_values('Otu') == oidx,:]
#     assemblages = thetasub.columns[(thetasub > otu_threshold).any(axis=0)]
#     return assemblages


# def get_taxa_in_assemblages(thetadf, assemblages, otu_threshold=0.05):
#     otus = thetadf.index[(thetadf.loc[:,assemblages]>otu_threshold).any(axis=1)].get_level_values('Otu')
#     return otus


# # TODO: move to utils
# def get_subj_averaged_assemblage_proportions(betadf):
#     subjave = betadf.loc[:,['Time','Assemblage','Value']].groupby(by=['Time','Assemblage']).mean()
#     subjave.reset_index(inplace=True)
#     betamat = subjave.pivot(index='Time', columns='Assemblage', values='Value')
#     return betamat


# def get_edge_weights(thetadf, avebeta, assemblages, otu_focus, otus_assoc):
#     times = avebeta.index
#     assemblages = thetadf.columns
#     thetasimp = thetadf.reset_index()[['Otu'] + list(assemblages)].set_index('Otu')
    
#     theta_self = thetasimp.loc[otu_focus,:].values # 1xK
#     theta_assoc = thetasimp.loc[otus_assoc,:].values # OxK
#     betavals = avebeta.loc[:,assemblages].values # ntime x K
    
#     numerator = np.sum(betavals[:,None,:]*theta_assoc*theta_self, axis=-1)
#     denominator = np.sum(betavals[:,None,:]*theta_self, axis=-1)
    
#     edgevals = numerator/denominator
#     df = pd.DataFrame(edgevals, index=times, columns=otus_assoc)
#     return df.T


# def get_relative_abundances(data, times, subjects, taxonomy, multi_index=False):
#     reads = data['count_data']
#     ntime = len(times)
#     nsubj = len(subjects)
#     notus = reads[times[0]][subjects[0]].shape[1]

#     relabuns = np.zeros((notus, ntime, nsubj)) # also make into dataframe
#     for i,t in enumerate(times):
#         for j,s in enumerate(subjects):
#             counts = reads[t][s].cpu().detach().clone().numpy()
#             pra = counts/counts.sum(axis=1,keepdims=True)
#             ras = np.mean(pra, axis=0)
#             relabuns[:,i,j] = ras

#     if multi_index is True:
#         index = pd.MultiIndex.from_frame(taxonomy.reset_index())
#     else:
#         index = taxonomy.index 
#     radf = pd.DataFrame(relabuns.mean(axis=2), index=index, columns=times)
#     return radf


# def plot_association_changes(axs, edge_weights, node_weights, otu_focus, taxonomy,
#                              node_colors=None, edge_scale=20, node_scale=2000, rad=1, textsize=12,
#                             edge_threshold=0.02, edge_base=0):
#     taxa = edge_weights.index
#     diets = edge_weights.columns

#     # make graph
#     graph = nx.Graph()
#     # add nodes and edges with default weights
#     for oidx in taxa:
#         if oidx != otu_focus:
#             graph.add_edge(oidx, otu_focus, weight=1)
#         graph.nodes[oidx]['weight'] = 500

#     # get node positions
#     pos = nx.spring_layout(graph)

#     notus = len(edge_weights.index)
#     xpos =  np.cos(np.linspace(0,1,notus)*2*np.pi)
#     ypos =  np.sin(np.linspace(0,1,notus)*2*np.pi)
#     i = 0
#     for oidx in edge_weights.index:
#         if oidx != otu_focus:
#             pos[oidx] = np.array([xpos[i],ypos[i]])
#             i += 1
#         else:
#             pos[oidx] = np.array([0,0])

#     for i,diet in enumerate(diets):
#         # update edge weights
#         for oidx in taxa:
#             if oidx != otu_focus:
#                 if edge_weights.loc[oidx,diet] >= edge_threshold:
#                     graph[oidx][otu_focus]['weight'] = edge_scale*edge_weights.loc[oidx,diet] + edge_base
#                 else:
#                     graph[oidx][otu_focus]['weight'] = 0
#             # update node sizes
#             graph.nodes[oidx]['weight']=node_scale*node_weights.loc[oidx,diet]

#         # get node colors
#         color_map = []
#         for node in graph:
#             if node_colors is not None:
#                 color_map.append(node_colors[node])
#             else:
#                 color_map.append('tab:blue')

#         # draw network
#         ew = list(nx.get_edge_attributes(graph,'weight').values())
#         nw = list(nx.get_node_attributes(graph,'weight').values())
#         nx.draw_networkx(graph, pos=pos, ax=axs[i], width=ew, node_size=nw, 
#                          node_color=color_map, with_labels=False, alpha=0.5)
        
#         # annotate nodes
#         for k in pos.keys():
#             x = pos[k][0]
#             y = pos[k][1]
#             if k == otu_focus:
#                 x = 1.0 - rad
#                 y = rad - 1.0
#             s = f'{k}'
#             s = s[(s.find('u')+1):]
#             axs[i].text(rad*x,rad*y,s=s, horizontalalignment='center', verticalalignment='center', fontsize=textsize)
#             if i == 0:
#                 s = taxonomy.loc[k,'Genus']
#                 axs[i].text(rad*x,rad*y + (rad-1.0),s=s, horizontalalignment='center', verticalalignment='center', fontsize=textsize)
#     return axs, graph
