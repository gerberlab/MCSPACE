import numpy as np 
import pandas as pd 
import torch 
from pathlib import Path 

# TODO: pass study directly into trainer/model -- don't need intermediate methods
#* final data structure is a dict; need to output in such format

class DataSet:
    """
    class to load single dataset from files
    """
    pass


class PerturbationDataSet:
    """
    class to load data from a perturbation study from tsv files
    handle filtering and preprocssing
    interface with visualization to check data and plot/show data stats
    interface with data_utils to get data for inference (or have member function...)
    """
    def __init__(self, pre_perturb, post_perturb, comparator=None):
        #* each file in long format; or list of tables? decide on one, can have utils to convert between if we want
        #* long format has columns for: otu, particle id, counts, subject-id
        #* currently enforcing same number of subjects for each group (this may not be necessary; but makes
        # data structures for inference easier...)
        #* ^ can discuss with georg how general we want package to be...
        self.pre_perturb = pd.read_csv(pre_perturb, sep="\t")
        self.post_perturb = pd.read_csv(post_perturb, sep="\t")
        if comparator is not None:
            self.comparator = pd.read_csv(comparator, sep="\t")
        else:
            self.comparator = None

        self.data, self.otu_index, self.num_subjects = self._process_data()
        
        
    def _process_data(self):
        # TODO: !!!
        #! assuming comparator group is given for now; might allow it not to be present later...
        
        # check each group has the same number of subjects
        nsubj_pre = len(self.pre_perturb.loc[:,"Subject"].unique())
        nsubj_post = len(self.post_perturb.loc[:,"Subject"].unique())
        nsubj_comp = len(self.comparator.loc[:,"Subject"].unique())
        
        if not (nsubj_pre == nsubj_post == nsubj_comp):
            # TODO: can we (do we want to?) relax this assumption?
            raise ValueError("Each group must contain the same number of subjects")
        else:
            nsubj = nsubj_pre
            
        def _get_otu_labels():
            pre_perturb_otus = set(self.pre_perturb.loc[:,'OTU'].unique())
            post_perturb_otus = set(self.post_perturb.loc[:,'OTU'].unique())
            comparator_otus = set(self.comparator.loc[:,'OTU'].unique())
            allotus = pre_perturb_otus.union(post_perturb_otus).union(comparator_otus)
            return list(allotus)
        
        def _get_group_data(df, otus):
            subjs = df.loc[:,"Subject"].unique()
            grpdata = {}
            for s in subjs:
                print(f'processing subject {s}')
                subjdf = df.loc[df.loc[:,"Subject"] == s,:]
                subjtable = subjdf.pivot_table(values='Count', index='OTU', columns='Particle', aggfunc=np.sum, fill_value=0)
                # set common set of otu index
                subjtable = subjtable.reindex(otus, fill_value=0)
                countdata = subjtable.values.T
                grpdata[s] = countdata 
            return grpdata

        index = _get_otu_labels() # get common set of otus for all groups
        data = {}
        # TODO: add logs/ output to get sense of if its working...
        print("processing pre-perturbed data")
        data['pre_perturb'] = _get_group_data(self.pre_perturb, index)
        print("processing post-perturbed data")
        data['post_perturb'] = _get_group_data(self.post_perturb, index)
        print("processing comparator data")
        data['comparator'] = _get_group_data(self.comparator, index)
    
        return data, pd.Index(index), nsubj
        
        
    def _print_group_stats(self, grp):
        npart_total = 0
        for subj in self.data[grp]:
            print(f"\t Subject {subj}:")
            counts = self.data[grp][subj]
            npart, notu = counts.shape
            npart_total += npart
            print(f"\t\t {npart} particles")
            rd = counts.sum(axis=1)
            print(f"\t\t min read depth: {np.amin(rd)}")
            print(f"\t\t median read depth: {np.median(rd)}")
            print(f"\t\t max read depth: {np.amax(rd)}")

        print(f"\t {npart_total} particles for group {grp}")
        
        
    def describe(self):
        """
        prints summary discription of current state of dataset
        """
            
        num_otus = len(self.otu_index)
        print("3 groups: pre-perturb, post-perturb, comparator")
        print(f"{num_otus} OTUs in study")
        print(f"{self.num_subjects} subjects per group")
        for grp in self.data:
            print(f"stats for {grp} group:")
            self._print_group_stats(grp)
            print("\n")
    
#     def get_data(self):
#         return self.data, self.otu_index
    

    def filter(self, min_abundance, min_reads, max_reads=None):
        """
        any kind of consistency? ...eg min abundance in at least x particles
        ...subject consistency? -- usual sort of otu filtering
        ...look for otu abundance in bulk?
        TODO: multiple options for filtering; see if we want to change/add others later
        TODO: add effects/reasoning/consequences for different filterings
        """
        #! using current filter from ICML paper for now
        # TODO: consider consitency filtering over subjects
        otu_subset = np.zeros(len(self.otu_index), dtype=bool)
        datadict = self.data
        
        for grp in datadict:
            for s in datadict[grp]:
                counts = datadict[grp][s] # P x O
                readdepth = counts.sum(axis=1)

                if min_reads > 0:
                    particle_threshold = min_reads
                else:
                    # TODO: remove this option???
                    particle_threshold = readdepth.sum()/2500 # default threshold
                    print(f"using particle threshold {particle_threshold} ..maybe remove variable option...")
                psub = (readdepth>particle_threshold)
                counts = counts[psub,:]

                bulk = counts.sum(axis=0)/counts.sum()
                otu_subset = otu_subset | (bulk > min_abundance)
    
        filtered_data = {}
        for grp in datadict:
            filtered_data[grp] = {}
            for s in datadict[grp]:
                counts = datadict[grp][s] # P x O

                filtered_counts = counts[:,otu_subset]
                readdepth = filtered_counts.sum(axis=1)

                if min_reads > 0:
                    particle_threshold = min_reads
                else:
                    # TODO: remove this option???
                    particle_threshold = readdepth.sum()/2500 # default threshold
                    print(f"post-OTU: using particle threshold {particle_threshold} ..maybe remove variable option...")
                psub = (readdepth>particle_threshold)
                filtered_counts = filtered_counts[psub,:]        
                filtered_data[grp][s] = filtered_counts
        self.data = filtered_data
        self.otu_index = self.otu_index[otu_subset]
        