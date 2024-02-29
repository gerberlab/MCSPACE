import numpy as np
import torch
from pathlib import Path
from mcspace.dataset import DataSet
import pandas as pd


def get_basic_data(reads_in, device, dtype=torch.float):
    bulk = reads_in.sum(axis=0)/reads_in.sum()
    contamination_communities = torch.from_numpy(bulk).to(dtype=dtype, device=device)
    reads = torch.from_numpy(reads_in).to(dtype)
    norm = torch.sum(reads, dim=1)
    rel_data = torch.div(reads, norm.unsqueeze(1))
    z_data = torch.log(rel_data+0.0001)
    z_std, z_mean = torch.std_mean(z_data, dim=1)
    z_data = z_data - z_mean.unsqueeze(1)
    z_data = torch.div(z_data, z_std.unsqueeze(1))
    if z_data.isnan().any():
        raise ValueError("nan in normed data")
    return {'count_data': reads.to(device), 'normed_data': z_data.to(device), 'full_normed_data': z_data.to(device)}, contamination_communities


def get_data(reads_dict, device):
    # reads_dict is dict (times) of dict (subjects) of counts
    counts = {}
    contamination_communities = {}
    normed_data = {}
    # output L* x O; for all particles concatenated together
    full_normed_data = [] 
    for g in reads_dict.keys():
        subjs = reads_dict[g].keys()
        counts[g] = {}
        all_particles = None
        normed_data[g] = {}
        for s in subjs:
            subj_reads = reads_dict[g][s]
            counts[g][s] = torch.from_numpy(subj_reads).to(dtype=torch.float, device=device)
            data, _ = get_basic_data(subj_reads, device)
            normed_data[g][s] = data['normed_data']
            full_normed_data.append(data['normed_data'])
            if all_particles is None:
                all_particles = reads_dict[g][s]
            else:
                all_particles = np.concatenate([all_particles, reads_dict[g][s]], axis=0)
        bulk = all_particles.sum(axis=0)/all_particles.sum()
        contamination_communities[g] = torch.from_numpy(bulk).to(dtype=torch.float, device=device)
    combined_data = torch.cat(full_normed_data, dim=0)
    return {'count_data': counts, 'normed_data': normed_data, 'full_normed_data': combined_data} #, contamination_communities


def filter_dataset(reads, min_abundance=0.005, min_reads=1000,max_reads=10000):
    rd = reads.sum(axis=1)
    psub = ((rd>=min_reads) & (rd<=max_reads))
    filtered = reads[psub,:]

    # filter otus by relative abundance threshold
    bulk = filtered.sum(axis=0)/filtered.sum()     
    otu_sub = (bulk > min_abundance)
    ofiltered = filtered[:,otu_sub]

    # filter again by read depth threshold after removing otus
    rd = ofiltered.sum(axis=1)
    psub = ((rd>=min_reads) & (rd<=max_reads))
    final_filtered = ofiltered[psub,:]
    return final_filtered, otu_sub

    # rd = reads.sum(axis=1)
    # psub = ((rd>=min_reads) & (rd<=max_reads))
    # filtered = reads[psub,:]
    # bulk = filtered.sum(axis=1)/filtered.sum()     
    # otu_sub = (bulk > min_abundance)
    # rd = reads.sum(axis=1)
    # psub = ((rd>=min_reads) & (rd<=max_reads))
    # pfiltered = reads[psub,:]
    # bulk = pfiltered.sum(axis=0)/pfiltered.sum()     
    # otu_sub = (bulk > min_abundance)
    # ofiltered = reads[:,otu_sub]

    # rd2 = ofiltered.sum(axis=1)
    # psub2 = ((rd2>=min_reads) & (rd2<=max_reads))
    # filtered = ofiltered[psub2,:]

def get_interpersonal_subjects(rootpath = Path("./")):    
    datapath = rootpath / "data"
    rawdata = pd.read_csv(datapath / "Interpersonal Data for Gary.csv")
    subjects = rawdata['Core'].unique()
    return subjects


def get_interpersonal_data_subject(subject, min_abundance=0.005, min_reads=1000,max_reads=10000, rootpath = Path("./")):
    datapath = rootpath / "data"

    rawdata = pd.read_csv(datapath / "Interpersonal Data for Gary.csv")

    subject_options = ['H10', 'H18', 'H19', 'H1', 'H11TS4']
    if subject not in subject_options:
        raise ValueError(f"subject must be in {subject_options}")

    temp = rawdata.loc[rawdata['Core'] == subject,:]
    temppivot = temp.pivot(index='OTU', columns='Particle', values='Count')
    temppivot = temppivot.fillna(0)
    min_reads=1000
    max_reads=10000
    min_abundance = 0.01

    filtered = temppivot.values
    rd = filtered.sum(axis=0)
    psub = ((rd>=min_reads) & (rd<=max_reads))
    filtered = filtered[:,psub]
    bulk = filtered.sum(axis=1)/filtered.sum()  
    otu_sub = (bulk > min_abundance)
    otufiltered = filtered[otu_sub,:]
    rd = otufiltered.sum(axis=0)
    psub2 = ((rd>=min_reads) & (rd<=max_reads))
    finalfiltered = otufiltered[:,psub2].T
    return finalfiltered


def get_mixing_dataset(min_abundance=0.005, min_reads=1000,max_reads=10000, datapath = Path("./data")):
    mixtest = np.load(datapath / "fig1_data_FIXT.npy")
    reads, otu_sub = filter_dataset(mixtest, min_abundance=min_abundance, min_reads=min_reads, max_reads=max_reads)
    taxa = pd.read_csv(datapath / "taxonomy_FINAL_fig1.tsv", sep="\t")
    taxasub = taxa.iloc[otu_sub,:]
    taxonomy = condense_taxonomy(taxasub)
    counts = {}
    counts[0] = {}
    counts[0]['s1'] = reads
    return counts, taxonomy


def get_high_fat_dataset(min_abundance=0.005, min_reads=1000,max_reads=10000, datapath = Path("./data")):
    hfdata = np.load(datapath / "high_fat_fig4_FIXT.npy")
    reads, otu_sub = filter_dataset(hfdata, min_abundance=min_abundance, min_reads=min_reads, max_reads=max_reads)
    taxa = pd.read_csv(datapath / "taxonomy_FINAL_fig4.tsv", sep="\t")
    taxasub = taxa.iloc[otu_sub,:]
    taxonomy = condense_taxonomy(taxasub)
    counts = {}
    counts[0] = {}
    counts[0]['s1'] = reads
    return counts, taxonomy


def get_low_fat_dataset(min_abundance=0.005, min_reads=1000,max_reads=10000, datapath = Path("./data")):
    lfdata = np.load(datapath / "low_fat_fig4_FIXT.npy")
    reads, otu_sub = filter_dataset(lfdata, min_abundance=min_abundance, min_reads=min_reads, max_reads=max_reads)
    taxa = pd.read_csv(datapath / "taxonomy_FINAL_fig4.tsv", sep="\t")
    taxasub = taxa.iloc[otu_sub,:]
    taxonomy = condense_taxonomy(taxasub)
    counts = {}
    counts[0] = {}
    counts[0]['s1'] = reads
    return counts, taxonomy


def get_small_intestine_dataset(min_abundance=0.005, min_reads=1000,max_reads=10000, datapath = Path("./data")):
    data = np.load(datapath / "small_intestine_fig3_FIXT.npy")
    reads, otu_sub = filter_dataset(data, min_abundance=min_abundance, min_reads=min_reads, max_reads=max_reads)
    taxa = pd.read_csv(datapath / "taxonomy_FINAL_fig3.tsv", sep="\t")
    taxasub = taxa.iloc[otu_sub,:]
    taxonomy = condense_taxonomy(taxasub)
    counts = {}
    counts[0] = {}
    counts[0]['s1'] = reads
    return counts, taxonomy


def get_cecum_dataset(min_abundance=0.005, min_reads=1000,max_reads=10000, datapath = Path("./data")):
    data = np.load(datapath / "cecum_fig3_FIXT.npy")
    reads, otu_sub = filter_dataset(data, min_abundance=min_abundance, min_reads=min_reads, max_reads=max_reads)
    taxa = pd.read_csv(datapath / "taxonomy_FINAL_fig3.tsv", sep="\t")
    taxasub = taxa.iloc[otu_sub,:]
    taxonomy = condense_taxonomy(taxasub)
    counts = {}
    counts[0] = {}
    counts[0]['s1'] = reads
    return counts, taxonomy


def get_colon_dataset(min_abundance=0.005, min_reads=1000,max_reads=10000,datapath = Path("./data")):
    data = np.load(datapath / "colon_fig3_FIXT.npy")
    reads, otu_sub = filter_dataset(data, min_abundance=min_abundance, min_reads=min_reads, max_reads=max_reads)
    taxa = pd.read_csv(datapath / "taxonomy_FINAL_fig3.tsv", sep="\t")
    taxasub = taxa.iloc[otu_sub,:]
    taxonomy = condense_taxonomy(taxasub)
    counts = {}
    counts[0] = {}
    counts[0]['s1'] = reads
    return counts, taxonomy


def get_diet_dataset(min_abundance=0.005, min_reads=1000, max_reads=10000, subj_remove=['JX09', 'JX07', 'JX08'], num_consistent_subjects=1,rootpath = Path("./")):
    datapath = rootpath / "data" / "diet_timeseries_data"
    taxfile = datapath / "taxonomy.csv"
    countfile = datapath / "count_data.csv"

    #* create dataset object
    dataset = DataSet(countfile, taxfile)

    # remove subject with missing data
    dataset.remove_subjects(subj_remove)

    # filter otus
    # dataset.consistency_filtering(num_consistent_subjects=2)
    dataset.consistency_filtering(num_consistent_subjects=num_consistent_subjects, min_abundance=min_abundance, min_reads=min_reads, max_reads=max_reads)

    # filter out lactococcus
    taxonomy = dataset.get_taxonomy()
    Lactococcus_oidxs = list(taxonomy.loc[taxonomy['Genus'] == 'Lactococcus',:].index)
    dataset.remove_otus(Lactococcus_oidxs)

#     dataset.remove_times(times_remove)

    # filter particles
    dataset.filter_particle_data(min_reads=min_reads, max_reads=max_reads)

    #* return reads, number otus, times, and subjects...
    reads = dataset.get_reads()
    num_otus = len(dataset.otu_index)
    times = dataset.times
    num_subjects = len(dataset.subjects)
    subjects = dataset.subjects
    return reads, num_otus, times, subjects, dataset


#! -----------------get paired datasets with perturbations----------------------------
def condense_taxonomy(rawtaxa, threshold=50):
    taxa2 = rawtaxa.copy()
    taxa2.set_index('Unnamed: 0', inplace=True) #! generalize?
    ranks = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus']
    confidence = [f'{rank} confidence' for rank in ranks]
    
    for i, row in taxa2.iterrows():
        for rank, conf in zip(ranks, confidence):
            if taxa2.loc[i,conf] < 50:
                taxa2.at[i,rank] = 'na'
    taxaout = taxa2.loc[:,ranks]
    return taxaout


def filter_dataset_pair(readsA, readsB, min_abundance=0.005, min_reads=1000, max_reads=10000):
    def _get_particle_otu_subsets(reads, min_abundance, min_reads, max_reads):
        rd = reads.sum(axis=1)
        psub = ((rd>=min_reads) & (rd<=max_reads))
        filtered = reads[psub,:]

        # filter otus by relative abundance threshold
        bulk = filtered.sum(axis=0)/filtered.sum()     
        otu_sub = (bulk > min_abundance)
        ofiltered = filtered[:,otu_sub]
        return psub, otu_sub

    psubA, otu_subA = _get_particle_otu_subsets(readsA, min_abundance, min_reads, max_reads)
    psubB, otu_subB = _get_particle_otu_subsets(readsB, min_abundance, min_reads, max_reads)
    otu_sub = (otu_subA | otu_subB)

    readsA_filtered = readsA[psubA,:][:,otu_sub]
    readsB_filtered = readsB[psubB,:][:,otu_sub]

    def _filter_particles(reads, min_reads, max_reads):    
        # filter again by read depth threshold after removing otus
        rd = reads.sum(axis=1)
        psub = ((rd>=min_reads) & (rd<=max_reads))
        final_filtered = reads[psub,:]
        return final_filtered

    ffA = _filter_particles(readsA_filtered, min_reads, max_reads)
    ffB = _filter_particles(readsB_filtered, min_reads, max_reads)

    # make dict, 2 time points, 1 subject each
    counts = {}
    counts[0] = {}
    counts[0]['s1'] = ffA
    counts[1] = {}
    counts[1]['s1'] = ffB
    return counts, otu_sub


def get_lf_hf_pair_data(min_abundance=0.005, min_reads=1000,max_reads=10000,datapath = Path("./data")):    
    lfreads = np.load(datapath / "low_fat_fig4_FIXT.npy")
    hfreads = np.load(datapath / "high_fat_fig4_FIXT.npy")
    counts, otusub = filter_dataset_pair(lfreads, hfreads, min_abundance, min_reads, max_reads)
    #* load taxonomy, filter, and return
    taxa = pd.read_csv(datapath / "taxonomy_FINAL_fig4.tsv", sep="\t")
    taxasub = taxa.iloc[otusub,:]
    taxonomy = condense_taxonomy(taxasub)
    return counts, taxonomy


def get_cecum_colon_pair_data(min_abundance=0.005, min_reads=1000,max_reads=10000,datapath = Path("./data")):
    cecum_reads = np.load(datapath / "cecum_fig3_FIXT.npy")
    colon_reads = np.load(datapath / "colon_fig3_FIXT.npy")
    counts, otusub = filter_dataset_pair(cecum_reads, colon_reads, min_abundance, min_reads, max_reads)

    #* load taxonomy, filter, and return
    taxa = pd.read_csv(datapath / "taxonomy_FINAL_fig3.tsv", sep="\t")
    taxasub = taxa.iloc[otusub,:]
    taxonomy = condense_taxonomy(taxasub)
    return counts, taxonomy


if __name__ == "__main__":
    reads, taxonomy = get_colon_dataset()
    print(reads.keys())
    print(reads[0].keys())
    print(reads[0]['s1'].shape)
    print(taxonomy.shape)
