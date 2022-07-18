import numpy as np
from configparser import ConfigParser
from utils import create_dir
import scipy 
import scipy.stats
from scipy.stats import multivariate_normal, truncnorm
from scipy.special import softmax, logsumexp, loggamma
from utils import get_logger
import logging 
import pickle 
import os 
from tqdm import tqdm 
import visualization 
import matplotlib.pyplot as plt 


def mvn_logprob(x, mu, cov):
    var = multivariate_normal(mean=mu, cov=cov)
    return var.pdf(x)


def multinomial_logprob(x, p):
    """ 
    logprob of product of categoricals, or unnormalized multinomial
    """
    return np.sum(x*np.log(p + 1e-8), axis=-1)


def TruncatedNormal(loc=0.0, scale=1.0, low=0, high=np.inf, size=None):
    #* scale = standard deviation
    return truncnorm(a=(low-loc)/scale,
                    b=(high-loc)/scale,
                    loc=loc,
                    scale=scale).rvs(size=size) 


def TruncatedNormal_logprob(x, loc=0.0, scale=1.0, low=0, high=np.inf):
    #* scale = standard deviation
    return truncnorm.logpdf(x,
                        a=(low-loc)/scale,
                        b=(high-loc)/scale,
                        loc=loc,
                        scale=scale)


def InvGamma_logprob(x, shape, scale):
    # a = shape; b = scale = 1/rate
    return -loggamma(shape) - shape*np.log(scale) - (shape+1)*np.log(x) - scale*x


class MapseqTopicModel:
    def __init__(self, config_file):
        parser = ConfigParser()
        parser.read(config_file)

        self.n_samples = parser.getint('general', 'n_samples')
        self.n_burnin = parser.getint('general', 'n_burnin')
        self.embedding_dim = parser.getint('general', 'embedding_dim')
        self.num_types = parser.getint('general', 'num_types')

        self.outdir = parser.get('general', 'basepath')
        create_dir(self.outdir)
        self.tag = parser.get('general', 'tag')

        self.data_dir = parser.get('data', 'reads')

        # model hyperparameters
        # shape and scale parameters for particle radius gamma distribution
        # TODO: switch to log-normal; have default parameters setup 
        self.lambda_shape = 2.5
        self.lambda_scale = 1.0 

        # prior mean and covariance for otu and particle type locations
        self.mvn_prior_mean = np.zeros((self.embedding_dim,))
        ones = np.ones((self.embedding_dim,))
        self.mvn_prior_covariance = np.diag(ones)

        # concentration parameter for dirichlet prior on beta
        alpha = parser.getfloat('general', 'alpha', fallback=1.0)
        self.beta_concentration_alpha = alpha*np.ones((self.num_types,))

        # logger
        log_path = os.path.join(self.outdir, "log.txt")
        self.logger = get_logger(log_path, level=logging.DEBUG)

    def load_data(self):
        # TODO: test/train split
        # TODO: read in taxonomy
        reads = np.load(self.data_dir)
        self.data_reads_lo = reads 
        self.num_particles, self.num_otus = reads.shape
        self.reads_per_particle = np.sum(self.data_reads_lo, axis=1)

        # plot reads and save figure
        ax = visualization.visualize_particle_reads(self.data_reads_lo, normalize=False, labels=True)
        plt.savefig(os.path.join(self.outdir, "loaded_read_data.png"))
        plt.close() 

    def init_params_mcmc(self):
        # model parameters
        self.otu_locations = np.random.multivariate_normal(self.mvn_prior_mean, self.mvn_prior_covariance, size=self.num_otus)
        self.particle_type_locations = np.random.multivariate_normal(self.mvn_prior_mean, self.mvn_prior_covariance, size=self.num_types)

        # using one-hot representation for indicators z_l = z_lk
        self.particle_type_indicators = np.zeros((self.num_particles, self.num_types))
        self.particle_radii = np.ones((self.num_particles,))

        # type distribution parameter beta; init as uniform distribution
        self.beta = np.ones((self.num_types,))/self.num_types 

        # INIT particle types from uniform prior
        for lidx in range(self.num_particles):
            self.sample_particle_types_prior(lidx) 

    def sample_prior(self):
        # sample otu locations
        for oidx in range(self.num_otus):
            self.sample_otu_location_prior(oidx)

        # sample particle type locations 
        for pidx in range(self.num_types):
            self.sample_particle_type_location_prior(pidx)

        self.sample_beta_prior()

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
        self.particle_radii[lidx] = 1.0/np.random.gamma(shape=self.lambda_shape, scale=self.lambda_scale)

    def sample_beta_prior(self):
        self.beta = np.random.dirichlet(alpha=self.beta_concentration_alpha)
        
    def sample_particle_types_prior(self, lidx):
        self.particle_type_indicators[lidx,:] = np.random.multinomial(n=1, pvals=self.beta)

    def train(self):
        n_samples = self.n_samples

        # parameter traces
        # TODO: consolidate parameters and traces in combined model parameter data structure
        self.trace_otu_locations = np.zeros((n_samples, self.num_otus, self.embedding_dim))
        self.trace_particle_type_locations = np.zeros((n_samples, self.num_types, self.embedding_dim))
        self.trace_particle_type_indicators = np.zeros((n_samples, self.num_particles, self.num_types))
        self.trace_particle_radii = np.zeros((n_samples, self.num_particles))
        self.trace_beta = np.zeros((n_samples, self.num_types))
        
        self.trace_log_likelihood = np.zeros((n_samples, 1))
        self.trace_log_prior = np.zeros((n_samples, 1))

        self.trace_otu_location_prior = np.zeros((n_samples, 1))
        self.trace_particle_type_prior  = np.zeros((n_samples, 1))
        self.trace_beta_prior  = np.zeros((n_samples, 1))
        self.trace_particle_indicator_prior  = np.zeros((n_samples, 1))
        self.trace_radii_prior = np.zeros((n_samples, 1))

        self.trace_perplexity = np.zeros((n_samples, 1))

        for i in tqdm(range(0, n_samples)):
            self.logger.debug(f"\n\nSAMPLE {i}:")
            self.sample_posterior()

            self.add_traces(i)

        mcmc_samples = self.get_samples()
        savefile = os.path.join(self.outdir, "mcmc.pkl")
        with open(savefile, 'wb') as h:
            pickle.dump(mcmc_samples, h)

    def add_traces(self, i):
        self.trace_otu_locations[i,:] = self.otu_locations
        self.trace_particle_type_locations[i,:] = self.particle_type_locations
        self.trace_particle_type_indicators[i,:] = self.particle_type_indicators
        self.trace_particle_radii[i,:] = self.particle_radii
        self.trace_beta[i,:] = self.beta

        loglike = self.calc_current_log_likelihood() 
        w_prior, u_prior, beta_prior, z_prior, radii_prior, logprior = self.calc_current_log_prior()  
        
        self.trace_otu_location_prior[i,:] = w_prior
        self.trace_particle_type_prior[i,:] = u_prior
        self.trace_beta_prior[i,:] = beta_prior
        self.trace_particle_indicator_prior[i,:] = z_prior
        self.trace_radii_prior[i,:] = radii_prior
        
        perplexity = self.calc_current_perplexity()

        self.trace_log_likelihood[i,:] = loglike
        self.trace_log_prior[i,:] = logprior
        self.trace_perplexity[i,:] = perplexity

    def calc_current_log_likelihood(self):
        theta_prime = np.einsum("l, kd, od -> lko", 1.0/self.particle_radii, self.particle_type_locations, self.otu_locations, optimize=True)
        theta_lko = softmax(theta_prime, axis=2)
        loglike = 0
        # TODO: can optimize this
        for lidx in range(self.num_particles):
            u = np.argmax(self.particle_type_indicators[lidx,:])
            theta_o = theta_lko[lidx,u,:]
            loglike += np.sum(self.data_reads_lo[lidx,:]*np.log(theta_o + 1e-8))
        return loglike

    def calc_current_log_prior(self):
        w_prior = 0 
        for oidx in range(self.num_otus):
            w_prior += mvn_logprob(self.otu_locations[oidx,:], self.mvn_prior_mean, self.mvn_prior_covariance)
        
        u_prior = 0 
        for uidx in range(self.num_types):
            u_prior += mvn_logprob(self.particle_type_locations[uidx,:], self.mvn_prior_mean, self.mvn_prior_covariance)
        
        beta_prior = np.sum((self.beta_concentration_alpha-1)*np.log(self.beta + 1e-8))
        
        z_prior = 0 #* categorical logprob given beta, for each l
        for lidx in range(self.num_particles):
            z_prior += np.sum(self.particle_type_indicators[lidx,:]*np.log(self.beta + 1e-8))

        radii_prior = 0 
        for lidx in range(self.num_particles):
            radii_prior += InvGamma_logprob(self.particle_radii[lidx], shape=self.lambda_shape, scale=self.lambda_scale)

        return w_prior, u_prior, beta_prior, z_prior, radii_prior, (w_prior + u_prior + beta_prior + z_prior + radii_prior)

    def calc_current_perplexity(self):
        # TODO: implement
        return 0

    def get_samples(self):
        return {"otu_locations": self.trace_otu_locations, "particle_type_locations": self.trace_particle_type_locations,\
            "particle_type_indicators": self.trace_particle_type_indicators, "particle_radii": self.trace_particle_radii,\
                "beta": self.trace_beta, "log_likelihood": self.trace_log_likelihood, "log_prior": self.trace_log_prior, \
                    "perplexity": self.trace_perplexity,\
        "otu_location_prior": self.trace_otu_location_prior, "particle_type_prior": self.trace_particle_type_prior, \
            "beta_prior": self.trace_beta_prior, "particle_indicator_prior": self.trace_particle_indicator_prior, \
                "radii_prior": self.trace_radii_prior } 

    def sample_posterior(self):
        # sample otu locations
        for oidx in range(self.num_otus):
            self.sample_otu_location_posterior(oidx)

        # sample particle type locations 
        for pidx in range(self.num_types):
            self.sample_particle_type_location_posterior(pidx)

        # sample beta parameter 
        self.sample_beta_posterior()

        # sample particle type indicators
        for lidx in range(self.num_particles):
            self.sample_particle_types_posterior(lidx)

        # TODO: add option to turn on/off; switch to lognormal prior
        # sample particle radii
        # for lidx in range(self.num_particles):
        #     self.sample_particle_radius_posterior(lidx)

    def eval_log_like(self, w, u, l, b):
        log_beta_lk = np.log(b)[None,:]
        theta_prime = np.einsum("l, kd, od -> lko", 1.0/l, u, w, optimize=True)
        theta_lko = softmax(theta_prime, axis=2)
        log_multinomial_lk = multinomial_logprob(self.data_reads_lo[:,None,:], theta_lko)  #* broadcast over lk; sum over o
        marg_log_like_l = logsumexp(log_multinomial_lk+log_beta_lk, axis=1)
        return np.sum(marg_log_like_l)

    def sample_otu_location_posterior(self, oidx):
        # sample using metropolis-hastings step 
        old_loglike = self.eval_log_like(self.otu_locations, self.particle_type_locations,\
                                        self.particle_radii, self.beta)

        self.otu_location_proposal_scale = self.mvn_prior_covariance  
        # TODO: tune to get optimal acceptance [tune scalar or matrix???]
        proposal_wo = np.random.multivariate_normal(self.otu_locations[oidx,:], self.otu_location_proposal_scale)

        proposal = np.copy(self.otu_locations)
        proposal[oidx,:] = proposal_wo

        new_loglike = self.eval_log_like(proposal, self.particle_type_locations,\
                                        self.particle_radii, self.beta)

        old_logprior = mvn_logprob(self.otu_locations[oidx,:], self.mvn_prior_mean, self.mvn_prior_covariance)
        new_logprior = mvn_logprob(proposal_wo, self.mvn_prior_mean, self.mvn_prior_covariance)

        self.logger.debug(f"OTU: {oidx}, old log likelihood = {old_loglike} ")
        self.logger.debug(f"OTU: {oidx}, new log likelihood = {new_loglike} ")
        self.logger.debug(f"OTU: {oidx}, old log prior = {old_logprior} ")
        self.logger.debug(f"OTU: {oidx}, new log prior = {new_logprior} ")

        # TODO: track acceptance ratio; tune during burnin...
        
        # update if accepted
        log_accept_numerator = new_loglike + new_logprior
        log_accept_denominator = old_loglike + old_logprior
        
        # if accept update current parameter to new value
        u = np.log(np.random.rand())
        if u < (log_accept_numerator - log_accept_denominator):
            self.otu_locations = proposal 
            self.logger.debug(f"OTU: {oidx}, accepted w_o proposal")

    def sample_particle_type_location_posterior(self, pidx):
        # sample using metropolis-hastings steps
        old_loglike = self.eval_log_like(self.otu_locations, self.particle_type_locations,\
                                            self.particle_radii, self.beta)
        
        self.type_location_proposal_scale = self.mvn_prior_covariance  
        proposal_uk = np.random.multivariate_normal(self.particle_type_locations[pidx,:], self.type_location_proposal_scale)

        proposal = np.copy(self.particle_type_locations)
        proposal[pidx,:] = proposal_uk

        new_loglike = self.eval_log_like(self.otu_locations, proposal,\
                                        self.particle_radii, self.beta)

        old_logprior = mvn_logprob(self.particle_type_locations[pidx,:], self.mvn_prior_mean, self.mvn_prior_covariance)
        new_logprior = mvn_logprob(proposal_uk, self.mvn_prior_mean, self.mvn_prior_covariance)

        self.logger.debug(f"Type: {pidx}, old log likelihood = {old_loglike} ")
        self.logger.debug(f"Type: {pidx}, new log likelihood = {new_loglike} ")
        self.logger.debug(f"Type: {pidx}, old log prior = {old_logprior} ")
        self.logger.debug(f"Type: {pidx}, new log prior = {new_logprior} ")

        
        # update if accepted
        log_accept_numerator = new_loglike + new_logprior
        log_accept_denominator = old_loglike + old_logprior
        
        # if accept update current parameter to new value
        u = np.log(np.random.rand())
        if u < (log_accept_numerator - log_accept_denominator):
            self.particle_type_locations = proposal 
            self.logger.debug(f"Type: {pidx}, accepted u_k proposal")

    def sample_particle_radius_posterior(self, lidx):
        # sample using metropolis-hastings steps
        # TODO: only need to evaluate likelihood for single particle L, no need to do all (should just cancel from both sides)
        old_loglike = self.eval_log_like(self.otu_locations, self.particle_type_locations,\
                                            self.particle_radii, self.beta)
        
        # using truncated normal for proposal for now
        self.particle_radius_proposal_scale = 0.3
        proposal_lambda = TruncatedNormal(loc=self.particle_radii[lidx], scale=self.particle_radius_proposal_scale)

        proposal = np.copy(self.particle_radii)
        proposal[lidx] = proposal_lambda

        new_loglike = self.eval_log_like(self.otu_locations, self.particle_type_locations,\
                                        proposal, self.beta)

        old_logprior = InvGamma_logprob(self.particle_radii[lidx], shape=self.lambda_shape, scale=self.lambda_scale)
        new_logprior = InvGamma_logprob(proposal_lambda, shape=self.lambda_shape, scale=self.lambda_scale)

        #* nonsymmetric proposal terms for MH
        old_given_new_logprob = TruncatedNormal_logprob(self.particle_radii[lidx], loc=proposal_lambda, scale=self.particle_radius_proposal_scale)
        new_given_old_logprob = TruncatedNormal_logprob(proposal_lambda, loc=self.particle_radii[lidx], scale=self.particle_radius_proposal_scale)

        self.logger.debug(f"Particle: {lidx}, old log likelihood = {old_loglike} ")
        self.logger.debug(f"Particle: {lidx}, new log likelihood = {new_loglike} ")
        self.logger.debug(f"Particle: {lidx}, old log prior = {old_logprior} ")
        self.logger.debug(f"Particle: {lidx}, new log prior = {new_logprior} ")
        self.logger.debug(f"Particle: {lidx}, old given new log-prob = {old_loglike} ")
        self.logger.debug(f"Particle: {lidx}, new given old log-prob = {new_loglike} ")
        
        # update if accepted
        log_accept_numerator = new_loglike + new_logprior + old_given_new_logprob
        log_accept_denominator = old_loglike + old_logprior + new_given_old_logprob
        
        # if accept update current parameter to new value
        u = np.log(np.random.rand())
        if u < (log_accept_numerator - log_accept_denominator):
            self.particle_radii = proposal 
            self.logger.debug(f"Particle: {lidx}, accepted new radius proposal")

    def sample_beta_posterior(self):
        self.beta = np.random.dirichlet(alpha=self.beta_concentration_alpha + np.sum(self.particle_type_indicators, axis=0))

    def sample_particle_types_posterior(self, lidx):
        # gibbs sample, calc probs for each k=u
        # TODO: this can be optimized/vectorize over k 
        logprobs = np.zeros((self.num_types,))
        for k in range(self.num_types):
            z_proposal_lk = np.copy(self.particle_type_indicators)
            z_proposal_lk[lidx,:] = 0
            z_proposal_lk[lidx,k] = 1 

            log_like = self.calc_loglike_z(lidx, z_proposal_lk)
            log_prior = self.calc_logprior_z(z_proposal_lk)

            logprobs[k] = log_like + log_prior 
        
        # sample from logprobs 
        g = np.random.gumbel(size=self.num_types)
        max_ind = np.argmax(g + logprobs)
        self.particle_type_indicators[lidx,:] = 0
        self.particle_type_indicators[lidx,max_ind] = 1

    def calc_loglike_z(self, lidx, z_lk):
        theta_prime = np.einsum("l, kd, od -> lko", 1.0/self.particle_radii, self.particle_type_locations, self.otu_locations, optimize=True)
        theta_lko = softmax(theta_prime, axis=2)
        u = np.argmax(z_lk[lidx,:])
        theta_o = theta_lko[lidx,u,:]
        return np.sum(self.data_reads_lo[lidx,:]*np.log(theta_o + 1e-8))

    def calc_logprior_z(self, z_lk):
        n = np.sum(z_lk, axis=0)
        a = self.beta_concentration_alpha 
        return loggamma(np.sum(a)) - np.sum(loggamma(a)) + np.sum(loggamma(n+a)) - loggamma(np.sum(n+a))

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
