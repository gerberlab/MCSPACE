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
    # cbar_kws = dict(extend='min')
    cbar_kws = dict(use_gridspec=False, location="bottom", shrink=0.5)

    ax=sns.heatmap(thetaplot, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, linecolor=linecolor, linewidth=linewidth,
               xticklabels=xticklabels, yticklabels=yticklabels, square=square, cbar=cbar, cbar_kws=cbar_kws)
    return ax
