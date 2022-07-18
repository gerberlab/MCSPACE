import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 


def plot_otu_embedding(w, u, annotate=False, ax=None):
    if ax is None:
        fig, ax = plt.subplots() 

    notus, dim = w.shape 
    ntypes, _ = u.shape 

    if dim != 2:
        print("not implemented")
        return 
    
    ax.scatter(w[:,0], w[:,1], alpha=0.5)
    ax.scatter(u[:,0], u[:,1], alpha=0.5, marker='d')

    if annotate is True:
        for i in range(notus):
            ax.annotate(str(i), (w[i,0], w[i,1] + 0.2))
        for i in range(ntypes):
            ax.annotate("T" + str(i), (u[i,0], u[i,1] + 0.2))

    return ax 


def plot_particle_type_indicators(z, ax=None):
    if ax is None:
        fig, ax = plt.subplots() 

    ax = sns.heatmap(data=z, cmap="Greys", ax=ax)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    return ax 

def plot_matrix(theta, annotate=False, ax=None):
    if ax is None:
        fig, ax = plt.subplots() 

    ax = sns.heatmap(data=theta.T, cmap="Greys", ax=ax, annot=annotate)
    return ax 

def plot_particle_type_distribution(theta, annotate=False, ax=None):
    if ax is None:
        fig, ax = plt.subplots() 

    ax = sns.heatmap(data=theta.T, cmap="Greys", ax=ax, annot=annotate)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    return ax 


def visualize_particle_reads(reads, **kwargs):
    """
    reads: numpy array of shape (number particles) x (number otus) giving counts
    """
    fig, ax = plt.subplots()
    ax = sns.heatmap(data=reads.T, cmap='Greys', ax=ax)  
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    return ax 


def visualize_particle_reads(reads, normalize=True, ax=None, labels=False):
    """
    reads: numpy array of shape (number particles) x (number otus) giving counts
    """
    # calculate relative abundance
    if normalize is True:
        ra = reads/(np.sum(reads, axis=1, keepdims=True)) 
    else:
        ra = reads 
        
    # sort by composition for otu

    ax = sns.heatmap(data=ra.T, ax=ax, cmap='Greys', cbar_kws = dict(use_gridspec=False,location="bottom"))  
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    
    if labels is False:
        ax.set_xticks([])
        ax.set_yticks([])
    return ax 


def plot_trace(trace_var, title=None, burnin=0, axs=None, marker_size=3, alpha=0.6):
    if axs is None:
        fig, ax = plt.subplots(ncols=2, constrained_layout=True)
    else:
        if len(axs) != 2:
            raise ValueError("need two axes for trace plot")
        ax = axs 

    msize = marker_size
    salpha = alpha

    var = trace_var[burnin:]

    ttl = title    
    # if title is not None:
    #     ttl = title + f"(med={med})"
    # else:
    #     ttl = f"(med={med})"

    t = np.arange(burnin, len(trace_var))
    ax[0].hist(var, alpha=0.5)
    ax[0].set_xlabel("Value")
    ax[0].set_ylabel("Probability")
    ax[0].set_title(ttl)
    ax[1].scatter(t, var, s=msize, alpha=salpha)
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Value")
    ax[1].set_title(ttl)

    return ax 


def plot_trace_2D(trace_var, burnin=0, ax=None, marker='o', alpha=0.5, label=None):
    if ax is None:
        fig, ax = plt.subplots()

    var = trace_var[burnin:,:]

    ax.scatter(var[:,0], var[:,1], alpha=alpha, label=label, marker=marker)

    return ax 


def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax.get_figure(), ax
    