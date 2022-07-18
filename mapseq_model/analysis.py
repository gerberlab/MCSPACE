import pickle 
import pandas as pd
import numpy as np 
import os
import skbio 
import skbio.stats.distance
from torch import threshold
import visualization
from scipy.special import softmax, comb
from utils import create_dir
import matplotlib.pyplot as plt 
import matplotlib 
matplotlib.use("agg")


# TODO:
# generalize w/radius
# rhat calc for multiple chains
# perplexity on held-out data
# topic coherence measures

class Analysis:
    def __init__(self, basepath, runpath, ground_truth=False, outpath=None):
        if outpath is None:
            self.outpath = os.path.join(runpath, "post_analysis")
        else:
            self.outpath = outpath
        create_dir(self.outpath)

        self.basepath = basepath 
        self.runpath = runpath 
        self.ground_truth = ground_truth

        mcmcpath = os.path.join(runpath, "mcmc.pkl") 

        with open(mcmcpath, 'rb') as handle:
            self.mcmc_samples = pickle.load(handle)

        # take map estimate 
        likelihood_trace = self.mcmc_samples["log_likelihood"]
        prior_trace = self.mcmc_samples["log_prior"]

        posterior_trace = np.squeeze(likelihood_trace + prior_trace)
        self.map_ind = np.argmax(posterior_trace)

        # return {"otu_locations": self.trace_otu_locations, "particle_type_locations": self.trace_particle_type_locations,\
        # "particle_type_indicators": self.trace_particle_type_indicators, "particle_radii": self.trace_particle_radii,\
        # "beta": self.trace_beta, "log_likelihood": self.trace_log_likelihood, "log_prior": self.trace_log_prior, \
        # "perplexity": self.trace_perplexity,\
        # "otu_location_prior": self.trace_otu_location_prior, "particle_type_prior": self.trace_particle_type_prior, \
        # "beta_prior": self.trace_beta_prior, "particle_indicator_prior": self.trace_particle_indicator_prior, \
        # "radii_prior": self.trace_radii_prior, } 

    def plot_mcmc_traces(self):
        mcmc_samples = self.mcmc_samples
        outpath = self.outpath 

        # plot likelihood and prior traces 
        likelihood_trace = mcmc_samples["log_likelihood"]
        fig, ax = plt.subplots() 
        ax = visualization.plot_trace(likelihood_trace, title="Log likelihood", burnin=0)
        plt.savefig(os.path.join(outpath, "log_likelihood_trace.png"))
        plt.close() 

        prior_trace = mcmc_samples["log_prior"]
        fig, ax = plt.subplots() 
        ax = visualization.plot_trace(prior_trace, title="Log prior", burnin=0)
        plt.savefig(os.path.join(outpath, "log_prior_trace.png"))
        plt.close() 

        otu_location_prior_trace = mcmc_samples["otu_location_prior"]
        fig, ax = plt.subplots() 
        ax = visualization.plot_trace(otu_location_prior_trace, title="Log OTU embed prior", burnin=0)
        plt.savefig(os.path.join(outpath, "otu_location_prior_trace.png"))
        plt.close() 

        particle_type_prior_trace = mcmc_samples["particle_type_prior"]
        fig, ax = plt.subplots() 
        ax = visualization.plot_trace(particle_type_prior_trace, title="Log type embed prior", burnin=0)
        plt.savefig(os.path.join(outpath, "particle_type_prior_trace.png"))
        plt.close() 

        radii_prior_trace = mcmc_samples["radii_prior"]
        fig, ax = plt.subplots() 
        ax = visualization.plot_trace(radii_prior_trace, title="Log radii prior", burnin=0)
        plt.savefig(os.path.join(outpath, "radii_prior_trace.png"))
        plt.close() 

        beta_prior_trace = mcmc_samples["beta_prior"]
        fig, ax = plt.subplots() 
        ax = visualization.plot_trace(beta_prior_trace, title="Log beta prior", burnin=0)
        plt.savefig(os.path.join(outpath, "beta_prior_trace.png"))
        plt.close() 

        particle_indicator_prior_trace = mcmc_samples["particle_indicator_prior"]
        fig, ax = plt.subplots() 
        ax = visualization.plot_trace(particle_indicator_prior_trace, title="Log indicator prior", burnin=0)
        plt.savefig(os.path.join(outpath, "particle_indicator_prior_trace.png"))
        plt.close()  

    def plot_learned_embedding(self):
        # plot embedding of map estimate
        otu_trace = self.mcmc_samples["otu_locations"]
        ptype_trace = self.mcmc_samples["particle_type_locations"]
        ax = visualization.plot_otu_embedding(otu_trace[self.map_ind,:], ptype_trace[self.map_ind,:], annotate=True)
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        ax.set_title("Embedding (MAP estimate)")
        plt.savefig(os.path.join(self.outpath, "map_embedding.png"))
        plt.close() 
 
    def plot_compare_radii_distributions(self):
        # look at radii distribution in map esimate {over particles}
        rad_trace = self.mcmc_samples["particle_radii"]
        rad_map = rad_trace[self.map_ind,:]

        plt.figure()
        plt.hist(rad_map, alpha=0.5, label='inferred')
        if self.ground_truth is True:
            rad_gt = np.load(os.path.join(self.basepath, "particle_radii.npy"))
            plt.hist(rad_gt, alpha=0.5, label='ground truth')
            plt.legend()
        plt.xlabel("Particle radius")
        plt.ylabel("Count")
        plt.title("Radius distribution (MAP estimate)")
        plt.savefig(os.path.join(self.outpath, "map_radii_dist.png"))
        plt.close()  

    def plot_compare_beta_distributions(self):
        # TODO: can also calculate a KL divergence... 
        # NOTE: ordering can be ambiguous; want to order properly...

        # look at beta and z in map estimate {compare with ground truth}
        beta_trace = self.mcmc_samples["beta"]
        beta_map = beta_trace[self.map_ind,:]

        x = [(i+1) for i in range(len(beta_map))]

        plt.figure()
        plt.bar(x, beta_map, alpha=0.5, label='inferred (MAP)')
        if self.ground_truth is True:
            beta_gt = np.load(os.path.join(self.basepath, "beta.npy"))
            x_gt = [(i+1) for i in range(len(beta_gt))]
            plt.bar(x_gt, beta_gt, alpha=0.5, label='ground truth')
            plt.legend()
        plt.xlabel("Particle type")
        plt.ylabel("Probability")
        plt.title("Beta (MAP)")
        plt.savefig(os.path.join(self.outpath, "beta_MAP.png"))
        plt.close()

    def plot_compare_type_indicator_distributions(self):
        # TODO: also can take count and compare with ground truth...
        
        # particle_type_indicators
        z_trace =self.mcmc_samples["particle_type_indicators"]
        z_map = z_trace[self.map_ind,:]

        fig, ax = plt.subplots()
        ax = visualization.plot_particle_type_indicators(z_map, ax=ax)
        ax.set_xlabel("Particle type")
        ax.set_ylabel("Particle ID")
        ax.set_title("Particle type indicators (MAP)")
        plt.savefig(os.path.join(self.outpath, "particle_assignments_MAP.png"))
        plt.close() 

        #* plot number of particles assigned each type
        n_k = np.sum(z_map, axis=0)

        x = [(i+1) for i in range(len(n_k))]

        # TODO: may need to compare TYPE ordering...
        plt.figure()
        plt.bar(x, n_k, alpha=0.5, label='inferred (MAP)')
        if self.ground_truth is True:
            z_gt = np.load(os.path.join(self.basepath, "particle_type_indicators.npy"))
            nk_gt = np.sum(z_gt, axis=0)
            x_gt = [(i+1) for i in range(len(nk_gt))]
            plt.bar(x_gt, nk_gt, alpha=0.5, label='ground truth')
            plt.legend()
        plt.xlabel("Particle type")
        plt.ylabel("Number particles assigned")
        plt.savefig(os.path.join(self.outpath, "n_particles_assigned.png"))
        plt.close()

        # TODO: can compare directly particle assignments for
        # TODO: ... each individual particle...

    def plot_compare_theta_distributions(self):
        # look at theta in map estimate [take min, med, max radii]
        otu_trace = self.mcmc_samples["otu_locations"]
        ptype_trace = self.mcmc_samples["particle_type_locations"]
        otu_map = otu_trace[self.map_ind,:]
        ptype_map = ptype_trace[self.map_ind,:]
        rad_trace = self.mcmc_samples["particle_radii"]
        rad_map = rad_trace[self.map_ind,:]

        median_radius = np.median(rad_map)

        theta_prime = np.einsum("l, kd, od -> lko", 1.0/rad_map, ptype_map, otu_map, optimize=True)
        theta_lko = softmax(theta_prime, axis=2)

        med_theta_prime = (1.0/median_radius)*np.einsum("kd, od -> ko", ptype_map, otu_map, optimize=True)
        med_theta_ko = softmax(med_theta_prime, axis=1)

        amin = np.argmin(rad_map)
        amax = np.argmax(rad_map)

        #* print out radius statistics? [5, 50, 95...]

        fig, ax = plt.subplots() 
        ax = visualization.plot_particle_type_distribution(theta_lko[amin,:,:], annotate=True)
        ax.set_xlabel("Particle type")
        ax.set_ylabel("OTU")
        ax.set_title(f"MAP Types distribution (Min radius)")
        plt.savefig(os.path.join(self.outpath, f"theta_distribution_min_MAP.png"))
        plt.close()

        fig, ax = plt.subplots() 
        ax = visualization.plot_particle_type_distribution(theta_lko[amax,:,:], annotate=True)
        ax.set_xlabel("Particle type")
        ax.set_ylabel("OTU")
        ax.set_title(f"MAP Types distribution (Max radius)")
        plt.savefig(os.path.join(self.outpath, f"theta_distribution_max_MAP.png"))
        plt.close()

        fig, ax = plt.subplots() 
        ax = visualization.plot_particle_type_distribution(med_theta_ko, annotate=True)
        ax.set_xlabel("Particle type")
        ax.set_ylabel("OTU")
        ax.set_title(f"MAP Types distribution (Median radius)")
        plt.savefig(os.path.join(self.outpath, f"theta_distribution_median_MAP.png"))
        plt.close()

    def calc_cluster_coocurrance_matrix(self, theta_gt_ko, theta_map_ko):
        # NOTE: picking fixed radius here...
        # 1] assign each otu a ground truth label from theta_gt_ko 
        # use argmax to get cluster label
        num_types_gt, num_otus = theta_gt_ko.shape 
        num_types_inf, _ = theta_map_ko.shape 

        gt_labels = np.zeros((num_otus))  # assign label i to each otu 
        for oidx in range(num_otus):
            gt_labels[oidx] = int(np.argmax(theta_gt_ko[:,oidx]))

        # assign a cluster to each otu
        inf_labels = np.zeros((num_otus))
        for oidx in range(num_otus):
            inf_labels[oidx] = int(np.argmax(theta_map_ko[:,oidx]))

        # i,j-th entry = number of occurances of label i in inferred cluster j 
        co_mat = np.zeros((num_types_gt, num_types_inf))
        for i in range(num_types_gt):
            for j in range(num_types_inf):
                # count over otus
                temp = 0 
                for oidx in range(num_otus):
                    # increment if has label i and j
                    if (gt_labels[oidx] == i) and (inf_labels[oidx] == j):
                        temp += 1
                co_mat[i,j] = temp
        return co_mat 

    def calc_NMI(self):
        if self.ground_truth is False:
            print("cannot calc NMI w/o 'gold standard'")
            return 

        # use threshold to get 'hard clusterings'
        #* calculate ground truth labels
        gt_rad = np.load(os.path.join(self.basepath, "particle_radii.npy"))
        gt_uk = np.load(os.path.join(self.basepath, "particle_type_locations.npy"))
        gt_wo = np.load(os.path.join(self.basepath, "otu_locations.npy"))
        num_otus, ndim = gt_wo.shape 
        num_types, _ = gt_uk.shape

        gt_theta_prime = np.einsum("l, kd, od -> lko", 1.0/gt_rad, gt_uk, gt_wo, optimize=True)

        gt_theta_lko = softmax(gt_theta_prime, axis=2)
        gt_theta_ko = gt_theta_lko[0,:,:]  # TODO: generalize? all radii same currently

        #* estimate inferred labels 
        otu_trace = self.mcmc_samples["otu_locations"]
        ptype_trace = self.mcmc_samples["particle_type_locations"]
        otu_map = otu_trace[self.map_ind,:]
        ptype_map = ptype_trace[self.map_ind,:]
        rad_trace = self.mcmc_samples["particle_radii"]
        rad_map = rad_trace[self.map_ind,:]

        median_radius = np.median(rad_map)

        # theta_prime = np.einsum("l, kd, od -> lko", 1.0/rad_map, ptype_map, otu_map, optimize=True)
        # theta_lko = softmax(theta_prime, axis=2)
        med_theta_prime = (1.0/median_radius)*np.einsum("kd, od -> ko", ptype_map, otu_map, optimize=True)
        med_theta_ko = softmax(med_theta_prime, axis=1)

        mat_yc = self.calc_cluster_coocurrance_matrix(gt_theta_ko, med_theta_ko) 
    
        #* calculate NMI
        n_y = np.sum(mat_yc, axis=1)
        p_y = n_y/num_otus
        h_y = -np.sum(p_y*np.log(p_y + 1e-8))

        # c - learned clustering 
        n_c = np.sum(mat_yc, axis=0)
        p_c = n_c/num_otus
        h_c = -np.sum(p_c*np.log(p_c + 1e-8))

        # calculate conditional entropy -> mutual 
        n_yc = mat_yc
        p_yc = n_yc/n_c
        h_ycc = -np.sum(p_yc*np.log(p_yc + 1e-8), axis=0) # sum over y; for each c

        h_yc = np.sum(p_c*h_ycc)

        i_yc = h_y - h_yc
        self.NMI = 2*i_yc/(h_y + h_c)

    def calc_RI(self):
        if self.ground_truth is False:
            print("cannot calc RI w/o 'gold standard'")
            return  
        
        def myComb(a,b):
            return comb(a,b,exact=True)

        vComb = np.vectorize(myComb)

        def get_tp_fp_tn_fn(cooccurrence_matrix):
            tp_plus_fp = vComb(cooccurrence_matrix.sum(0, dtype=int),2).sum()
            tp_plus_fn = vComb(cooccurrence_matrix.sum(1, dtype=int),2).sum()
            tp = vComb(cooccurrence_matrix.astype(int), 2).sum()
            fp = tp_plus_fp - tp
            fn = tp_plus_fn - tp
            tn = comb(cooccurrence_matrix.sum(), 2) - tp - fp - fn

            return [tp, fp, tn, fn]

        #* calculate ground truth labels
        gt_rad = np.load(os.path.join(self.basepath, "particle_radii.npy"))
        gt_uk = np.load(os.path.join(self.basepath, "particle_type_locations.npy"))
        gt_wo = np.load(os.path.join(self.basepath, "otu_locations.npy"))
        num_otus, ndim = gt_wo.shape 
        num_types, _ = gt_uk.shape

        gt_theta_prime = np.einsum("l, kd, od -> lko", 1.0/gt_rad, gt_uk, gt_wo, optimize=True)

        gt_theta_lko = softmax(gt_theta_prime, axis=2)
        gt_theta_ko = gt_theta_lko[0,:,:]  # TODO: generalize? all radii same currently

        #* estimate inferred labels 
        otu_trace = self.mcmc_samples["otu_locations"]
        ptype_trace = self.mcmc_samples["particle_type_locations"]
        otu_map = otu_trace[self.map_ind,:]
        ptype_map = ptype_trace[self.map_ind,:]
        rad_trace = self.mcmc_samples["particle_radii"]
        rad_map = rad_trace[self.map_ind,:]

        median_radius = np.median(rad_map)

        # theta_prime = np.einsum("l, kd, od -> lko", 1.0/rad_map, ptype_map, otu_map, optimize=True)
        # theta_lko = softmax(theta_prime, axis=2)
        med_theta_prime = (1.0/median_radius)*np.einsum("kd, od -> ko", ptype_map, otu_map, optimize=True)
        med_theta_ko = softmax(med_theta_prime, axis=1)

        # TODO: what to do with variable radii?; overlapping clusters?
        co_mat = self.calc_cluster_coocurrance_matrix(gt_theta_ko, med_theta_ko)
        
        tp, fp, tn, fn = get_tp_fp_tn_fn(co_mat)
        self.RI = (tp + tn)/(tp + tn + fp + fn)

        print(f"RI = {self.RI}")

    def compute_cooccurance_probs_ground_truth(self):
        # calculate co-occurance probs p(i,j|lk) = p(i|lk)p(j|lk) since indep
        # marg over lk p(l)p(k) from posterior samples {from map}
        # p(i|lk) = theta_lki

        rad_gt = np.load(os.path.join(self.basepath, "particle_radii.npy"))
        z_gt = np.load(os.path.join(self.basepath, "particle_type_indicators.npy"))
        ptype_gt = np.load(os.path.join(self.basepath, "particle_type_locations.npy"))
        otu_gt = np.load(os.path.join(self.basepath, "otu_locations.npy"))
    
        theta_prime = np.einsum("l, kd, od -> lko", 1.0/rad_gt, ptype_gt, otu_gt, optimize=True)
        theta_lko = softmax(theta_prime, axis=2)

        num_particles, num_types, num_otus = theta_lko.shape 

        pij = np.zeros((num_otus, num_otus))  
        for i in range(num_otus):
            for j in range(num_otus):
                # marginalize over particle l and types k
                temp = 0 
                for lidx in range(num_particles):
                    for kidx in range(num_types):
                        temp += theta_lko[lidx,kidx,i]*theta_lko[lidx,kidx,j]*z_gt[lidx,kidx]
                pij[i,j] = temp/num_particles

        return pij 

    def compute_cooccurance_probs(self):
        # calculate co-occurance probs p(i,j|lk) = p(i|lk)p(j|lk) since indep
        # marg over lk p(l)p(k) from posterior samples {from map}
        # p(i|lk) = theta_lki
        otu_trace = self.mcmc_samples["otu_locations"]
        ptype_trace = self.mcmc_samples["particle_type_locations"]
        otu_map = otu_trace[self.map_ind,:]
        ptype_map = ptype_trace[self.map_ind,:]
        rad_trace = self.mcmc_samples["particle_radii"]
        rad_map = rad_trace[self.map_ind,:]

        # particle_type_indicators
        z_trace =self.mcmc_samples["particle_type_indicators"]
        z_map = z_trace[self.map_ind,:]

        theta_prime = np.einsum("l, kd, od -> lko", 1.0/rad_map, ptype_map, otu_map, optimize=True)
        theta_lko = softmax(theta_prime, axis=2)

        num_particles, num_types, num_otus = theta_lko.shape 

        pij = np.zeros((num_otus, num_otus))  
        for i in range(num_otus):
            for j in range(num_otus):
                # marginalize over particle l and types k
                temp = 0 
                for lidx in range(num_particles):
                    for kidx in range(num_types):
                        temp += theta_lko[lidx,kidx,i]*theta_lko[lidx,kidx,j]*z_map[lidx,kidx]
                pij[i,j] = temp/num_particles

        return pij 

    def calc_mantel(self):
        if self.ground_truth is False:
            print("cannot calc mantel w/o 'comparator'")
            return  

        # plot cooccurance probs
        copbs_map = self.compute_cooccurance_probs()
        copbs_gt = self.compute_cooccurance_probs_ground_truth()
        ax = visualization.plot_matrix(copbs_map)
        plt.savefig(os.path.join(self.outpath, f"cooccurance_probs_MAP.png"))
        plt.close()

        ax = visualization.plot_matrix(copbs_gt)
        plt.savefig(os.path.join(self.outpath, f"cooccurance_probs_GT.png"))
        plt.close()

        num_otus, _ = copbs_map.shape
        for i in range(num_otus):
            copbs_map[i,i] = 0
            copbs_gt[i,i] = 0

        # TODO: compare pearson vs spearman...
        coeff, pval, n = skbio.stats.distance.mantel(copbs_gt, copbs_map)
        fnorm = np.sqrt(np.sum((copbs_gt - copbs_map)**2))

        self.mantel_corr = coeff 
        self.mantel_pval = pval
        return coeff, pval, fnorm

    def compare_relative_abundance(self):
        # NOTE: do not need ground truth to check this
        # learned RA p(i) = marg p(i|lk)

        otu_trace = self.mcmc_samples["otu_locations"]
        ptype_trace = self.mcmc_samples["particle_type_locations"]
        otu_map = otu_trace[self.map_ind,:]
        ptype_map = ptype_trace[self.map_ind,:]
        rad_trace = self.mcmc_samples["particle_radii"]
        rad_map = rad_trace[self.map_ind,:]

        # particle_type_indicators
        z_trace =self.mcmc_samples["particle_type_indicators"]
        z_map = z_trace[self.map_ind,:]

        theta_prime = np.einsum("l, kd, od -> lko", 1.0/rad_map, ptype_map, otu_map, optimize=True)
        theta_lko = softmax(theta_prime, axis=2)

        num_particles, num_types, num_otus = theta_lko.shape 

        pi_map = np.zeros((num_otus,))  
        for i in range(num_otus):
            # marginalize over particle l and types k
            temp = 0 
            for lidx in range(num_particles):
                for kidx in range(num_types):
                    temp += theta_lko[lidx,kidx,i]*z_map[lidx,kidx]
            pi_map[i] = temp/num_particles

        # compute ground truth relative abundance
        #* get from data 
        r_lo = np.load(os.path.join(self.basepath, "particle_reads.npy"))

        relative_abundance_lo = r_lo/np.sum(r_lo, axis=1, keepdims=True)

        phat_i = np.sum(relative_abundance_lo, axis=0)/num_particles

        # plot dists on top of each other
        x = list(range(num_otus))
        plt.figure()
        plt.bar(x, phat_i, alpha=0.5, label="data")
        plt.bar(x, pi_map, alpha=0.5, label="MAP")
        plt.legend()
        plt.savefig(os.path.join(self.outpath, "relative_abundances.png"))
        plt.close() 

        # compute kl divergence measure wrt data
        kl_div = np.sum(pi_map*np.log(pi_map/phat_i))
        self.ra_kl_div = kl_div 

    def output_metrics(self):
        # output to csv file...
        nplaces = 3
        data = {'NMI': round(self.NMI, nplaces), 
                'RI': round(self.RI, nplaces), 
                'RA KL-Div': round(self.ra_kl_div, nplaces), 
                'Mantel corr': round(self.mantel_corr, nplaces), 
                'Mantel pval': round(self.mantel_pval, nplaces)
                }

        df = pd.DataFrame(data=data, index=[0])
        fig,ax = visualization.render_mpl_table(df, header_columns=0, col_width=2.0)        
        plt.savefig(os.path.join(self.outpath, "metrics_table.png"))
        plt.close() 


if __name__ == "__main__":
   

    # basepath = "./data/two_comm_half/"  # directory containing ground truth data
    # runpath = "./runs/test_run"  # directory containing test output

    # basepath = "./data/two_comm_overlap"
    # runpath = "./runs/two_comm_overlap"

    basepath = "./data/three_comm_one_uniform"
    runpath = "./runs/three_comm_one_uniform"

    ana = Analysis(basepath, runpath, ground_truth=True)

    ana.plot_mcmc_traces()
    ana.plot_learned_embedding()
    ana.plot_compare_radii_distributions()
    ana.plot_compare_beta_distributions()
    ana.plot_compare_type_indicator_distributions()
    # ana.plot_compare_theta_prime_distributions()
    ana.plot_compare_theta_distributions()
    ana.calc_NMI()
    ana.calc_RI()
    coeff, pval, fnorm = ana.calc_mantel()
    print(f"coeff = {coeff}")
    print(f"pval = {pval}")
    print(f"fnorm = {fnorm}")
    ana.compare_relative_abundance() #* save kl divergence measure
    
    ana.output_metrics()
    # output:
    # nmi, ri, rel-abun kl divergence, mantel test
    # TODO: kl-divergence between beta and z as well [but need ordering]
    print("***DONE w/POST ANALYSIS***")
