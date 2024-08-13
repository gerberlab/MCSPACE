import numpy as np 
import pandas as pd 
import torch 
from pathlib import Path 
import matplotlib.pyplot as plt
import seaborn as sns


class DataSet:
    """
    class to load single dataset from long format csv files
    """
    #* save this with model as pickle object, use for plotting downstream...
    def __init__(self, reads, taxonomy, gzip=False):
        self.reads_file = reads
        self.taxonomy_file = taxonomy

        self._raw_taxonomy = pd.read_csv(taxonomy)
        self._raw_taxonomy.set_index('Otu', inplace=True)
        if gzip is False:
            self._long_data = pd.read_csv(reads)
        else:
            self._long_data = pd.read_csv(reads, compression='gzip')
            
        self.times = np.sort(np.array(self._long_data['Time'].unique())) #TODO: add check to make sure this is numeric...
        self.subjects = self._long_data['Subject'].unique()
        
        self._raw_data = self._get_raw_data()
        self._raw_otu_index = self._get_common_otu_index()

        self.otu_index = self._raw_otu_index.copy()
        self.reads = None

    def _get_raw_data(self):
        rawdata = {}
        for tm in self.times:
            rawdata[tm] = {}
            for subj in self.subjects:
                temp = self._long_data.loc[(self._long_data['Subject'] == subj) & (self._long_data['Time'] == tm),:]
                temppivot = temp.pivot(index='OTU', columns='Particle', values='Count')
                tempfinal = temppivot.fillna(0)
                rawdata[tm][subj] = tempfinal
        return rawdata

    def _get_common_otu_index(self):
        rolling_total_idx = set()
        for tm in self.times:
            for subj in self.subjects:
                temp = self._raw_data[tm][subj]
                rolling_total_idx = rolling_total_idx.union(set(temp.index))
        return list(rolling_total_idx)

    def remove_times(self, to_remove):
        self.times = np.sort(np.setdiff1d(self.times, np.array(to_remove)))

    def remove_subjects(self, to_remove):
        self.subjects = np.setdiff1d(self.subjects, np.array(to_remove))

    def consistency_filtering(self, num_consistent_subjects, min_abundance=0.005, min_reads=1000, max_reads=10000):
        # TODO: add consistency over time points too??
        taxa_presence = dict() # dict over days
        for tm in self.times:
            otu_sub_group = dict()
            for subj in self.subjects:        
                combined_reps = []
                temp = self._raw_data[tm][subj]
                temp = temp.reindex(self._raw_otu_index, fill_value=0)
                # filter particles
                filtered = temp.values # O x L
                rd = filtered.sum(axis=0)
                psub = ((rd>=min_reads) & (rd<=max_reads))
                filtered = filtered[:,psub]
                bulk = filtered.sum(axis=1)/filtered.sum()     
                otu_sub_group[subj] = (bulk > min_abundance)
            taxa_presence[tm] = pd.DataFrame(otu_sub_group, index=self._raw_otu_index)

        otus_keep = set()
        for tm in self.times:
            temp = taxa_presence[tm]
            otu_keep = (temp.sum(axis=1) >= num_consistent_subjects)
            otus_in_day = set(otu_keep[otu_keep].index)
            otus_keep = otus_keep.union(otus_in_day)
        self.otu_index = list(otus_keep)

    def remove_otus(self, to_remove):
        for oidx in to_remove:
            self.otu_index.remove(oidx) 

    def filter_particle_data(self, min_reads=1000, max_reads=10000):
        data = {} # dict over time - subject
        for tm in self.times:
            data[tm] = {}
            for subj in self.subjects:
                temp = self._raw_data[tm][subj]
                # filter otus and reindex
                temp = temp.reindex(self.otu_index, fill_value=0)
                # filter particles
                filtered = temp.values # O x L
                rd = filtered.sum(axis=0)
                psub = ((rd>=min_reads) & (rd<=max_reads))
                filtered = filtered[:,psub]
                data[tm][subj] = filtered.T
        self.reads = data 

    def get_taxonomy(self):
        return self._raw_taxonomy.loc[self.otu_index,:]

    def get_reads(self):
        if self.reads is None:
            raise ValueError("need to perform particle filtering...")
        else:
            return self.reads

    def plot_relative_abundance(self, subject, ax, taxlevel="Family", topN=20):
        # TODO: implement topN and taxlevel...; and subject...
        taxonomy = self.get_taxonomy()
        multiind = pd.MultiIndex.from_frame(taxonomy)

        # create relative abundance dataframes
        dfs = []
        notus = len(self.otu_index)
        nsub = len(self.subjects)
        for tm in self.times:
            ra = np.zeros((nsub,notus))
            for i,sub in enumerate(self.subjects):
                temp = self.reads[tm][sub]
                rabun = temp.sum(axis=0)/temp.sum()
                ra[i,:] = rabun
            dftemp = pd.DataFrame(data=ra.T, index=multiind, columns=self.subjects)
            dfs.append(dftemp)

        n_time = len(self.times)
        FONTSIZE = 16
        fig, ax = plt.subplots(figsize=(25,5), ncols=n_time, sharey=True)
        for i,grp in enumerate(self.times):
            # diet = diets[i]
            temp = dfs[i]
            grouped = temp.groupby(level=['Order','Family']).sum()
            grouped.T.plot(kind='bar',stacked=True, ax=ax[i], cmap='tab20')
            if i < 6:
                ax[i].get_legend().remove()
            else:
                ax[i].legend(bbox_to_anchor=(1.01,1.01), title="(Order, Family)")
            ax[i].set_title(f"Day {grp}", fontsize=FONTSIZE)
            ax[i].set_xlabel("Subject", fontsize=FONTSIZE)
            ax[i].set_xticklabels(ax[i].get_xticklabels(), fontsize=FONTSIZE)
        ax[0].set_yticks([0,0.2,0.4,0.6,0.8,1.0])
        ax[0].set_yticklabels(ax[0].get_yticklabels(), fontsize=FONTSIZE)
        # plt.savefig(outpath / "diet_pert_data_RD1000.png", bbox_inches="tight")
        return fig, ax

    def get_data_stats(self):
        # TODO: might want to spit out values somehow instead -- option to return dataframe?
        # TODO RENAME COLUMNS...
        rd_stats = {}
        rd_stats['times'] = []
        rd_stats['subjs'] = []
        rd_stats['mean_rd'] = []
        rd_stats['std_rd'] = []
        rd_stats['min_rd'] = []
        rd_stats['rd25'] = []
        rd_stats['rd50'] = []
        rd_stats['rd75'] = []
        rd_stats['max_rd'] = []
        rd_stats['npart'] = [] 

        for tm in self.times:
            for sub in self.subjects:
                temp = self.reads[tm][sub]
                rds = temp.sum(axis=1)
                rd_stats['times'].append(tm)
                rd_stats['subjs'].append(sub)
                rd_stats['npart'].append(temp.shape[0])
                rd_stats['mean_rd'].append(np.mean(rds))
                rd_stats['std_rd'].append(np.std(rds))
                rd_stats['min_rd'].append(np.amin(rds))
                rd_stats['max_rd'].append(np.amax(rds))
                rd_stats['rd25'].append(np.percentile(rds, q=25))
                rd_stats['rd50'].append(np.percentile(rds, q=50))
                rd_stats['rd75'].append(np.percentile(rds, q=75))

        rdstatsdf = pd.DataFrame(rd_stats)
        return rdstatsdf
