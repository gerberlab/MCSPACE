import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from mcspace.comparators.utils import ilr_transform_data, inv_ilr_transform_data
from mcspace.comparators.EM_intercept_inference import EM_algo as EM_onedim
from mcspace.comparators.EM_intercept_inference import calc_expected_states as one_dim_calc_expected_states
from mcspace.comparators.EM_twodirection_inference import calc_expected_states as two_dim_calc_expected_states
from mcspace.comparators.EM_twodirection_inference import EM_algo as EM_twodim
"""
wrappers for comparator models: sklearn's GMM, and directional GMM models (1D and 2D)
"""


class BasicGaussianMixture:
    def __init__(self, num_communities):
        self.num_communities = num_communities
        self.model = GaussianMixture(n_components=self.num_communities)

    def get_params(self):
        # get aic, learned communities (means), weights, covariances, convergence check, number-iterations used
        # TODO: for perturbations, need some type of map? particle assignment...
        params = {}
        params['labels'] = self.labels
        params['aic'] = self.aic
        params['weights'] = self.model.weights_
        params['means'] = self.model.means_
        params['covariances'] = self.model.covariances_
        params['converged'] = self.model.converged_
        params['n_iter'] = self.model.n_iter_
        params['theta'] = inv_ilr_transform_data(self.model.means_)
        return params

    def fit_model(self, data):
        # TODO: what is format of input data?; transform to what's needed...
        ilr_data = ilr_transform_data(data)
        self.labels = self.model.fit_predict(ilr_data)
        self.aic = self.model.aic(ilr_data)


    def predict_labels(self, data):
        ilr_data = ilr_transform_data(data)
        pred = self.model.predict(ilr_data)
        return pred
    
    def get_communities(self):
        mu = inv_ilr_transform_data(self.model.means_)
        return mu


# #! run range of K and get aics
# def run_basic_gmm_vs_k(data, k_list):
#     aics = []
#     params = []
#     for k in k_list:
#         model = BasicGaussianMixture(k)
#         model.fit_model(data)
#         p = model.get_params()
#         aics.append(p['aic'])
#         params.append(p)
#     # TODO: check min aic; and make sure all converged...
#     #* return aics and params with lowest aic
#     return aics, params

def lognormpdf(y, mu, sigma):
    # y shape = LxO; mu shape = KxO, sigma shape = KxOxO
    # result shape = LxK
    sign, logdet = np.linalg.slogdet(sigma)
    # TODO: better way than calculating inverse?
    sigmainv = np.linalg.inv(sigma)
    x = y[:,None,:] - mu[None,:,:]
    prod = np.einsum('lki,kij,lkj->lk',x,sigmainv,x)
    k = y.shape[1]
    loglik_lk = -0.5*(logdet + prod + k*np.log(2*np.pi))
    return loglik_lk.sum()


def directional_AIC(model, y):
    mixing = model["mixing"]
    sigma = model["sigma"]
    mu = model["mu"]
    num_bins = len(mixing)
    log_likelihood=lognormpdf(y, mu, sigma)
    ntaxa = len(sigma[0])
    num_params = num_bins*((ntaxa**2-ntaxa)/2+2*ntaxa+1)-1
    return num_params*2-2*log_likelihood


class DirectionalGaussianMixture:
    def __init__(self, num_communities, dim):
        self.num_communities = num_communities
        #TODO: if 2-dim, enforce k is even; else throw error
        self.dim = dim
        if self.dim == 1:
            self.num_bins = self.num_communities
        elif self.dim == 2:
            self.num_bins = int(self.num_communities/2)
        else:
            raise ValueError("invalid dim")

    def get_aic(self, data):
        data_ILR = ilr_transform_data(data)
        return directional_AIC(self.results, data_ILR)
        
    def get_params(self):
        return {'results': self.results,
                'dim': self.dim,
                'ncomm': self.num_communities,
                'aic': self.aic,
                'labels': self.labels,
                'means': self.results['mu'],
                'theta': inv_ilr_transform_data(self.results['mu'])}

    def fit_model(self, data):
        data_ILR = ilr_transform_data(data)
        res = {}
        if self.dim == 1:
            model = EM_onedim(data_ILR, num_bins=self.num_bins)
            mixing, sigma, delta, Q, Q_edge, edge_mean, mu, likelihoods, iterations = model
            res['mixing'] = mixing
            res['sigma'] = sigma
            res['delta'] = delta
            res['Q'] = Q
            res['Q_edge'] = Q_edge
            res['edge_mean'] = edge_mean
            res['mu'] = mu
            res['likelihoods'] = likelihoods
            res['iterations'] = iterations
        elif self.dim == 2:
            model = EM_twodim(data_ILR, K=self.num_bins)
            mixing, sigma, delta, delta_perp, Q, Q_edge, edge_mean, mu, likelihoods, iterations = model
            res['mixing'] = mixing
            res['sigma'] = sigma
            res['delta'] = delta
            res['delta_perp'] = delta_perp
            res['Q'] = Q
            res['Q_edge'] = Q_edge
            res['edge_mean'] = edge_mean
            res['mu'] = mu
            res['likelihoods'] = likelihoods
            res['iterations'] = iterations
        self.results = res
        self.aic = self.get_aic(data)
        self.labels = self.predict_labels(data)

    def predict_labels(self, data):
        mu = self.results['mu']
        sigma = self.results['sigma']
        mixing = self.results['mixing']

        if len(data.shape) == 1:
            data = data[None,:]
        nsamples = data.shape[0]
        dataset = ilr_transform_data(data)
        if self.dim == 1:
            z = one_dim_calc_expected_states(self.num_bins, nsamples, dataset, mu, sigma, mixing)
        if self.dim == 2:
            z = two_dim_calc_expected_states(self.num_bins, nsamples, dataset, mu, sigma, mixing)
        return np.argmax(z, axis=1)

    def get_communities(self):
        mu = inv_ilr_transform_data(self.results['mu'])
        return mu
    