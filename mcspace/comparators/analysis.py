import numpy as np
from mcspace.comparators.utils import inv_ilr_transform_data
from mcspace.utils import hellinger_distance


def get_gmm_community_reconstruction_error(params, gt_theta):
    learned_communities = inv_ilr_transform_data(params['means'])
    
