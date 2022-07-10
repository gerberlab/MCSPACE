import numpy as np
from scipy.special import softmax 


class MapseqTopicModel:
    def __init__(self, num_otus, num_particles, num_types, embedding_dim):
        self.num_otus = num_otus
        self.num_particles = num_particles
        self.num_types = num_types
        self.embedding_dim = embedding_dim

        # model parameters
        self.otu_locations = np.zeros((self.num_otus, self.embedding_dim))
        self.particle_type_locations = np.zeros((self.num_types, self.embedding_dim))

        # using one-hot representation
        self.particle_type_indicators = np.zeros((self.num_particles, self.num_types)) 
        self.particle_radii = np.zeros((self.num_particles,))

        # model hyperparameters
        # shape and scale parameters for particle radius gamma distribution
        self.lambda_shape = 1.0  
        self.lambda_scale = 0.1 

        # concentration parameter for dirichlet prior on beta
        self.beta_concentration_alpha = (1/self.num_types)*np.ones((self.num_types,))
        
        # reads per particle # TODO: to get from data
        self.reads_per_particle = 100*np.ones((self.num_particles))

        # prior mean and covariance for otu and particle type locations
        self.mvn_prior_mean = np.zeros((self.embedding_dim,))
        ones = np.ones((self.embedding_dim,))
        self.mvn_prior_covariance = np.diag(ones)

    def sample_prior(self):
        # sample otu locations
        for oidx in range(self.num_otus):
            self.sample_otu_location_prior(oidx)

        # sample particle type locations 
        for pidx in range(self.num_types):
            self.sample_particle_type_location_prior(pidx)

        # sample distribution of particle types
        for lidx in range(self.num_particles):
            self.sample_particle_types_prior(lidx)

        # sample particle radii
        for lidx in range(self.num_particles):
            self.sample_particle_radius_prior(lidx)

    def sample_otu_location_prior(self, oidx):
        self.otu_locations[oidx,:] = np.random.multivariate_normal(self.mvn_prior_mean, self.mvn_prior_covariance)

    def sample_particle_type_location_prior(self, pidx):
        self.particle_type_locations[pidx,:] = np.random.multivariate_normal(self.mvn_prior_mean, self.mvn_prior_covariance)

    def sample_particle_radius_prior(self, lidx):
        self.particle_radii[lidx] = np.random.gamma(shape=self.lambda_shape, scale=self.lambda_scale)

    def sample_particle_types_prior(self, lidx):
        beta = np.random.dirichlet(alpha=self.beta_concentration_alpha)
        self.particle_type_indicators[lidx,:] = np.random.multinomial(n=1, pvals=beta)

    def sample_posterior(self):
        # sample otu locations
        for oidx in range(self.num_otus):
            self.sample_otu_location_posterior(oidx)

        # sample particle type locations 
        for pidx in range(self.num_types):
            self.sample_particle_type_location_posterior(pidx)

        # sample particle type indicators
        for lidx in range(self.num_particles):
            self.sample_particle_types_posterior(lidx)

        # sample particle radii
        for lidx in range(self.num_particles):
            self.sample_particle_radius_posterior(lidx)

    def sample_otu_location_posterior(self, oidx):
        pass 

    def sample_particle_type_location_posterior(self, pidx):
        pass

    def sample_particle_radius_posterior(self, lidx):
        pass 

    def sample_particle_types_posterior(self, lidx):
        pass 

    def get_theta_distribution(self, particle_radius=None):
        if particle_radius is None:
            theta_prime = np.einsum("l, kd, od -> lko", 1.0/self.particle_radii, self.particle_type_locations, self.otu_locations, optimize=True)
            theta = softmax(theta_prime, axis=2)
        else:
            theta_prime = (1/particle_radius)*np.einsum("kd, od -> ko", self.particle_type_locations, self.otu_locations, optimize=True)
            theta = softmax(theta_prime, axis=1)
        return theta 

    def generate_reads(self):
        reads = np.zeros((self.num_particles, self.num_otus)) 
        theta_prime = np.einsum("l, kd, od -> lko", 1.0/self.particle_radii, self.particle_type_locations, self.otu_locations, optimize=True)
        theta = softmax(theta_prime, axis=2)

        for lidx in range(self.num_particles):
            k = np.argmax(self.particle_type_indicators[lidx,:])
            reads[lidx,:] = np.random.multinomial(n=self.reads_per_particle[lidx], pvals=theta[lidx,k,:])

        return reads
