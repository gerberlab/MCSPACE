import numpy as np
from sklearn.mixture import GaussianMixture
from mcspace.comparators.utils import ilr_transform_data
from mcspace.comparators.EM_intercept_inference import EM_algo as EM_onedim
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
        return params

    def fit_model(self, data):
        # TODO: what is format of input data?; transform to what's needed...
        ilr_data = ilr_transform_data(data)
        self.labels = self.model.fit_predict(ilr_data)
        self.aic = self.model.aic(ilr_data)


#! run range of K and get aics
def run_basic_gmm_vs_k(data, k_list):
    aics = []
    params = []
    for k in k_list:
        model = BasicGaussianMixture(k)
        model.fit_model(data)
        p = model.get_params()
        aics.append(p['aic'])
        params.append(p)
    # TODO: check min aic; and make sure all converged...
    #* return aics and params with lowest aic
    return aics, params

class DirectionalGaussianMixture:
    def __init__(self, num_communities, dim):
        pass
#* if 2-dim, enforce k is even; else throw error

    def get_aic(self, data):
        pass

    def get_params(self):
        pass

    def fit_data(self, data):
        # self.model(data)# ...
        # self.get_aic(data) ...
        pass