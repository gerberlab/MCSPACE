import numpy as np 
import pandas as pd 
import torch 
from pathlib import Path 
import matplotlib.pyplot as plt
import seaborn as sns

# TODO: add documentation on main class/functions and their argument options

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

    def get_particle_stats(self): #, reads, times, subjects):
        ptms = []
        psubjs = []
        num_particles = []

        for t in self.times:
            for s in self.subjects:
                counts = self.reads[t][s]
                npart = counts.shape[0]
                
                ptms.append(t)
                psubjs.append(s)
                num_particles.append(npart)
                
        npart_df = pd.DataFrame({'Time': ptms, 'Subject': psubjs, 'Number particles': num_particles})
        return npart_df

    def get_read_stats(self):
        tms = []
        subjs = []
        num_reads = []
        particle_id = []

        for t in self.times:
            for s in self.subjects:
                counts = self.reads[t][s]
                npart = counts.shape[0]
                nreads_all = counts.sum(axis=1)
                
                for lidx in range(npart):
                    tms.append(t)
                    subjs.append(s)
                    particle_id.append(lidx)
                    num_reads.append(nreads_all[lidx])
        nreads_df = pd.DataFrame({'Time': tms, 'Subject': subjs, 'Particle': particle_id, 'Number reads': num_reads})
        return nreads_df
    
    def get_relative_abundances(self):
        taxonomy = self.get_taxonomy()
        num_otus = taxonomy.shape[0]
        multiind = pd.MultiIndex.from_frame(taxonomy)
        ntime = len(self.times)
        
        radfs = {}
        for s in self.subjects:
            ra = np.zeros((ntime, num_otus))
            for i,t in enumerate(self.times):
                counts = self.reads[t][s]
                rabun = counts/counts.sum(axis=1, keepdims=True)
                bulk_rabun = np.mean(rabun, axis=0)
                ra[i,:] = bulk_rabun
            df = pd.DataFrame(data=ra.T, index=multiind, columns=self.times)
            radfs[s] = df
        return radfs # separate dataframe for each subject
    