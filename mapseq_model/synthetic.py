import visualization
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils import create_dir 
from scipy.special import softmax
import os 

np.random.seed(1)

#* pipeline
# a few established cases/topologies by creating latent embedding 
# forward simulate model from fixed parameters ^
# visualize and output data

#* established cases:
# two communities half split 
# two communities plus homogeneous 
# three communities with overlap #? how exactly?

class Synthetic:
    def __init__(self, outpath):
        self.outpath = outpath 
        create_dir(self.outpath)

    # Example cases
    # ============================================================================================================
    def two_community_half_split(self):
        self.num_otus = 12 
        self.num_particles = 200 
        self.num_types = 2 
        self.embedding_dim = 2 
        self.reads_per_particle = 1000*np.ones((self.num_particles,))

        w_o = np.zeros((self.num_otus, self.embedding_dim))
        for oidx in range(self.num_otus):
            theta = (oidx%2)*(2*np.pi/2)
            w_o[oidx,0] = np.cos(theta) + 0.1*(np.random.rand()-0.5)
            w_o[oidx,1] = np.sin(theta) + 0.1*(np.random.rand()-0.5)

        u_k = np.ones((self.num_types, self.embedding_dim))
        for i in range(self.num_types):
            theta_u = i*(2*np.pi/2)
            u_k[i,0] = np.cos(theta_u)
            u_k[i,1] = np.sin(theta_u)

        self.beta = np.array([0.8, 0.2])
        self.otu_locations = w_o
        self.particle_type_locations = u_k 
        self.particle_radii = np.ones((self.num_particles,))

    def two_community_with_overlap(self):
        self.num_otus = 12 
        self.num_particles = 200 
        self.num_types = 2 
        self.embedding_dim = 2 
        self.reads_per_particle = 1000*np.ones((self.num_particles,))

        w_o = np.zeros((self.num_otus, self.embedding_dim))
        for oidx in range(self.num_otus - 4):  #* leave last 4 in center; part of both communities
            theta = (oidx%2)*(2*np.pi/2)
            w_o[oidx,0] = np.cos(theta) + 0.1*(np.random.rand()-0.5)
            w_o[oidx,1] = np.sin(theta) + 0.1*(np.random.rand()-0.5)

        u_k = np.ones((self.num_types, self.embedding_dim))
        for i in range(self.num_types):
            theta_u = i*(2*np.pi/2)
            u_k[i,0] = np.cos(theta_u)
            u_k[i,1] = np.sin(theta_u)

        self.beta = np.array([0.8, 0.2])
        self.otu_locations = w_o
        self.particle_type_locations = u_k 
        self.particle_radii = np.ones((self.num_particles,))

    def three_comm_one_uniform(self):
        self.num_otus = 12 
        self.num_particles = 200 
        self.num_types = 3
        self.embedding_dim = 2 
        self.reads_per_particle = 1000*np.ones((self.num_particles,))

        w_o = np.zeros((self.num_otus, self.embedding_dim))
        for oidx in range(self.num_otus):  #* leave last 4 in center; part of both communities
            theta = (oidx%2)*(2*np.pi/2)
            w_o[oidx,0] = np.cos(theta) + 0.1*(np.random.rand()-0.5)
            w_o[oidx,1] = np.sin(theta) + 0.1*(np.random.rand()-0.5)

        u_k = np.zeros((self.num_types, self.embedding_dim))
        for i in range(1, self.num_types):
            theta_u = i*(2*np.pi/2)
            u_k[i,0] = np.cos(theta_u)
            u_k[i,1] = np.sin(theta_u)

        self.beta = np.array([0.6, 0.2, 0.2]) #* sample more from first [uniform]
        self.otu_locations = w_o
        self.particle_type_locations = u_k 
        self.particle_radii = np.ones((self.num_particles,))


    # forward simulate and save
    # ============================================================================================================
    def simulate_dataset(self):
        # sample particle types
        self.particle_type_indicators = np.zeros((self.num_particles, self.num_types))
        for lidx in range(self.num_particles):
            self.particle_type_indicators[lidx,:] = np.random.multinomial(n=1, pvals=self.beta)

        # generate reads 
        self.reads = np.zeros((self.num_particles, self.num_otus)) 
        theta_prime = np.einsum("l, kd, od -> lko", 1.0/self.particle_radii, self.particle_type_locations, self.otu_locations, optimize=True)
        self.theta = softmax(theta_prime, axis=2)

        for lidx in range(self.num_particles):
            k = np.argmax(self.particle_type_indicators[lidx,:])
            self.reads[lidx,:] = np.random.multinomial(n=self.reads_per_particle[lidx], pvals=self.theta[lidx,k,:])

        self.plot_data()

        # output ground truth data
        np.save(os.path.join(self.outpath, "particle_reads.npy"), self.reads)
        np.save(os.path.join(self.outpath, "beta.npy"), self.beta)
        np.save(os.path.join(self.outpath, "otu_locations.npy"), self.otu_locations)
        np.save(os.path.join(self.outpath, "particle_type_locations.npy"), self.particle_type_locations)
        np.save(os.path.join(self.outpath, "particle_radii.npy"), self.particle_radii)
        np.save(os.path.join(self.outpath, "particle_type_indicators.npy"), self.particle_type_indicators)
        
        print(f"data output to <{self.outpath}>")

    def plot_data(self):
        fig, ax = plt.subplots() 
        ax = visualization.plot_otu_embedding(self.otu_locations, self.particle_type_locations, annotate=True, ax=ax)
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        xmin = np.amin(np.concatenate([self.otu_locations[:,0], self.particle_type_locations[:,0]]))
        xmax = np.amax(np.concatenate([self.otu_locations[:,0], self.particle_type_locations[:,0]]))
        ymin = np.amin(np.concatenate([self.otu_locations[:,1], self.particle_type_locations[:,1]]))
        ymax = np.amax(np.concatenate([self.otu_locations[:,1], self.particle_type_locations[:,1]]))
        ax.set_xlim(xmin-0.2, xmax+0.2)
        ax.set_ylim(ymin-0.2, ymax+0.2)
        ax.set_title("OTU embedding")
        plt.savefig(os.path.join(self.outpath, "otu_embedding.png"))
        plt.close()

        # plot radii
        radii = self.particle_radii 
        fig, ax = plt.subplots()
        ax.hist(radii)
        ax.set_xlabel("Radius")
        ax.set_ylabel("Count")
        ax.set_title("Particle radii distribution")
        plt.savefig(os.path.join(self.outpath, "particle_radii.png"))
        plt.close()

        beta = self.beta 
        x = [(i+1) for i in range(len(beta))]
        plt.figure()
        plt.bar(x, beta)
        plt.xlabel("Particle type")
        plt.ylabel("Probability")
        plt.savefig(os.path.join(self.outpath, "beta.png"))
        plt.close()

        # plot particle assignment
        z_lk = self.particle_type_indicators
        fig, ax = plt.subplots()
        ax = visualization.plot_particle_type_indicators(z_lk, ax=ax)
        ax.set_xlabel("Particle type")
        ax.set_ylabel("Particle ID")
        ax.set_title("Particle type indicators")
        plt.savefig(os.path.join(self.outpath, "particle_assignments.png"))
        plt.close()

        # plot theta for min and max radii
        theta_lko = self.theta
        amin = np.argmin(radii)
        amax = np.argmax(radii)

        fig, ax = plt.subplots() 
        ax = visualization.plot_particle_type_distribution(theta_lko[amin,:,:], annotate=True)
        ax.set_xlabel("Particle type")
        ax.set_ylabel("OTU")
        ax.set_title("Types distribution (Min radius)")
        plt.savefig(os.path.join(self.outpath, "theta_distribution_min.png"))
        plt.close()

        fig, ax = plt.subplots() 
        ax = visualization.plot_particle_type_distribution(theta_lko[amax,:,:], annotate=True)
        ax.set_xlabel("Particle type")
        ax.set_ylabel("OTU")
        ax.set_title("Types distribution (Max radius)")
        plt.savefig(os.path.join(self.outpath, "theta_distribution_max.png"))
        plt.close()

        reads = self.reads
        # plot reads and relative abundances...
        fig, ax = plt.subplots()
        ax = visualization.visualize_particle_reads(reads, normalize=False, labels=True, ax=ax) 
        ax.set_xlabel("Particle ID")
        ax.set_ylabel("OTU")
        ax.set_title("Particle reads")
        plt.savefig(os.path.join(self.outpath, "particle_reads.png"))
        plt.close()


if __name__ == "__main__":
    # outpath = "./data/two_comm_half"
    # outpath = "./data/two_comm_overlap"
    outpath = "./data/three_comm_one_uniform"
    synthetic = Synthetic(outpath) 
    # synthetic.two_community_half_split()
    # synthetic.two_community_with_overlap()
    synthetic.three_comm_one_uniform()
    synthetic.simulate_dataset()
    