import numpy as np
from scipy.stats import multivariate_normal, invwishart
from sklearn.cluster import KMeans
import copy
from sklearn.decomposition import PCA
import itertools
from .basic_GMM_EM import EM_algo as basic_EM_algo
import sys

def set_trace(A, target):
    """
    A is a matrix, target is what we want A's trace to be
    """
    tr = np.trace(A)
    w, v = np.linalg.eigh(A)
    w = w*target/tr
    eig_mat = np.zeros((len(w), len(w)))
    for i in range(len(w)):
        eig_mat[i,i]=w[i]
    #eig mat is now the new diagonal matrix
    return v.dot(eig_mat).dot(v.T)

def rearrange_means(mu):
    """
    Rearrange bin means to minimize total distance between them
    """
    def dist(mu_perm):
        dist=0
        num_bins=len(mu_perm)
        for i in range(num_bins-1):
            dist+=np.linalg.norm(mu_perm[i+1]-mu_perm[i])
        return dist
    perms = list(itertools.permutations([i for i in range(len(mu))]))
    best_dist=None
    best_mu = copy.deepcopy(mu)
    new_mu = copy.deepcopy(mu)
    best_perm = None
    for i in range(len(perms)):
        for j in range(len(mu)):
            new_mu[j] = mu[perms[i][j]]
        if best_dist is None or dist(new_mu)<best_dist:
            best_dist = dist(new_mu)
            best_mu = copy.deepcopy(new_mu)
            best_perm = perms[i]
    return best_mu, best_perm

def initialize_params(dataset, num_bins, num_dim, ilr_flag=True):
    """
    Parameters
    ----------
    num_bins : Int
        Number of Bins
    num_dim : Int
        Number of effective taxa (ntaxa-1 for philr)
    Returns
    -------
    Initialized params for mixing param,noise covariances, spatial covariance, 
    edge mean, and edge covariance, and all bin means
    
    Note: I also show alternative initialization methods in this code
    """
    

    mixing = np.zeros(num_bins)
    sigma = np.zeros((num_bins, num_dim, num_dim))
    for i in range(num_bins):
        mixing[i] = 1/num_bins
        sigma[i] = 0.1*np.eye(num_dim)
    iw = invwishart(df=num_dim+1, scale=np.identity(num_dim))
    
    sigma = 0.1*iw.rvs((num_bins,))
    Q_edge = iw.rvs((1,))
    Q_edge = set_trace(Q_edge, 0.1)
    
    Q = iw.rvs((1,))
    Q = set_trace(Q, 0.1)

    #One option for delta init: Use pca largest component
    pca = PCA(n_components=1)
    pca.fit(dataset)
    delta = pca.components_[0]*pca.singular_values_[0] 
    
    #Initializing means with KMeans
    mu = np.zeros((num_bins, num_dim))
    basic_kmeans = KMeans(n_clusters=num_bins).fit(dataset) #We DO NOT set the random state- want a different initialization everytime 
    mu = basic_kmeans.cluster_centers_
    mu, perm = rearrange_means(mu)
    
    #What the algo actually uses: initialize with the naive GMM
    fitted = basic_EM_algo(dataset, num_bins, threshold=1e-3)
    basic_mixing, basic_sigma, basic_mu, likelihoods, iterations = fitted
    mu = copy.deepcopy(basic_mu)
    mu, perm = rearrange_means(mu)
    for i in range(len(perm)):
        sigma[i] = basic_sigma[perm[i]]
        mixing[i] = basic_mixing[perm[i]]

    edge_mean = mu[0]

    ntaxa = num_dim
    new_delta = np.zeros(ntaxa)
    for k in range(1, num_bins):
        mu_diff = mu[k]-mu[k-1]
        new_delta += mu_diff
    delta = new_delta/(num_bins-1) #delta is the average distance between the initial means
    new_Q = np.zeros((ntaxa, ntaxa))
    for k in range(1, num_bins):
        mu_diff = mu[k]-mu[k-1]-delta
        mu_diff = np.reshape(mu_diff, (-1, 1))
        new_Q += np.dot(mu_diff, mu_diff.T)
    new_Q /= (num_bins-1)

    new_Q = bound_eigenval(new_Q)
    #print("new Q condition number: " + str(np.linalg.cond(new_Q)))
    w, v = np.linalg.eig(new_Q)
    #print("new Q eigenvalues: " + str(w))
    Q = np.copy(new_Q)

    return mixing, sigma, delta, Q, Q_edge, edge_mean, mu

def calc_l_likelihood(expected_states, nsamples, num_bins, dataset, mixing, mu, sigma, edge_mu, edge_Q, Q, delta):
    log_likelihood = 0
    for k in range(num_bins):
        log_likelihood += np.sum(expected_states[:,k]*multivariate_normal.logpdf(dataset, mu[k], sigma[k]))
        log_likelihood += np.log(mixing[k])*np.sum(expected_states[:,k])

    log_likelihood += multivariate_normal.logpdf(mu[0], edge_mu.flatten(), edge_Q)

    for k in range(1, num_bins):
        log_likelihood += multivariate_normal.logpdf(mu[k], mu[k-1]+delta, Q) 

    return log_likelihood

def calc_expected_states(num_bins, nsamples, dataset, mu, sigma, mixing):
    #Performing the E-step in the EM algorithm
    probabilities = np.zeros((num_bins, nsamples)) #log probability of bin assignment
    for i in range(num_bins):
        probabilities[i,:] = multivariate_normal.logpdf(dataset, mu[i], sigma[i])
    expected_states = np.zeros((nsamples, num_bins))
    for n in range(nsamples):
        x = np.log(mixing)+probabilities[:,n] #Will use logsumexp to deal with numerical underflow
        x = np.array(x)
        a = np.amax(x)
        denom = a+np.log(np.sum(np.exp(x-a)))
        for k in range(num_bins):
            expected_states[n,k] = np.exp(x[k]-denom)
    return expected_states

def bound_eigenval(A):
    """
    Deals with numerical stability by lower bounding all eigenvalues of A at a normal threshold (1e-3)
    Parameters
    ----------
    A : positive definite covariance matrix
    Returns
    -------
    recalculated A

    """
    w, v = np.linalg.eigh(A)
    eig_mat = np.zeros((len(w), len(w)))
    for i in range(len(w)):
        if w[i] < 1e-3:
            w[i]=1e-3
        if isinstance(w[i], complex):
            w[i]=1e-3
        eig_mat[i,i]=w[i]
    #eig mat is now the new diagonal matrix
    return v.dot(eig_mat).dot(v.T)

def EM_algo(dataset, num_bins):
    """
    Parameters
    ----------
    dataset : 2D np array
        number of samples x number of taxa
    num_bins: int
        Number of mixtures to learn
    Returns
    -------
    Returns model params, likelihood from each step, and number of iterations requireds
    """
    def E_Sigma(expectation_states, sigma, k):
        sig_inv = np.linalg.inv(sigma[k])
        return np.sum(expectation_states[:,k])*sig_inv
    ntaxa = dataset.shape[1]
    nsamples = dataset.shape[0]
    finished_flag = False
    #Just initializing them so it's easy to return them outside of the loop
    mixing, sigma, delta, Q, Q_edge, edge_mean, mu = [None for i in range(7)]

    likelihoods = []
    iterations = 0
    attempts = 0
    while finished_flag is False:
        print("ATTEMPT")
        try:
            likelihoods = []
            iterations = 0
            mixing, sigma, delta, Q, Q_edge, edge_mean, mu = initialize_params(dataset, num_bins, ntaxa)
            edge_mean = np.reshape(edge_mean, (-1, 1))
            prev_l_likelihood=-0.01
            expected_states = calc_expected_states(num_bins, nsamples, dataset, mu, sigma, mixing)
            
            in_row = 0 #counts how many times in a row the likelihood has changed less than the threshold

            for max_iterations in range(500): 
                expected_states = calc_expected_states(num_bins, nsamples, dataset, mu, sigma, mixing)
                
                #Implementation of Thomas Algorithm for updating the means
                Q_inv = np.linalg.inv(Q)
                high_off_diag = [Q_inv for i in range(num_bins-1)]
                low_off_diag = copy.deepcopy(np.array(high_off_diag))


                diag = np.zeros((num_bins, ntaxa, ntaxa))
                diag[0] = -np.linalg.inv(Q_edge)-high_off_diag[0]-E_Sigma(expected_states, sigma, 0)
                for i in range(1, num_bins):
                    diag[i] = -2*high_off_diag[0]-E_Sigma(expected_states, sigma,i)
                diag[num_bins-1]+=high_off_diag[0] #last element has a different expression
                #Must also calculate the gradient
                grad = np.zeros((num_bins, ntaxa))
                for k in range(num_bins):
                    sig_inv = np.linalg.inv(sigma[k])
                    for n in range(nsamples):
                        grad[k]+=(expected_states[n,k]*np.dot(sig_inv, dataset[n]))
            
                grad[0] += np.dot(np.linalg.inv(Q_edge), edge_mean).flatten()
                grad[0] -= np.dot(Q_inv, delta).flatten()
                grad[num_bins-1] += np.dot(Q_inv, delta)
                grad*=-1
                gamma = np.zeros((num_bins-1, ntaxa, ntaxa))
                gamma[0] = np.matmul(np.linalg.inv(diag[0]), high_off_diag[0])
                for i in range(1, num_bins-1):
                    gamma[i] = np.matmul(np.linalg.inv(diag[i]-np.matmul(low_off_diag[i-1],gamma[i-1])),high_off_diag[i])
                beta = np.zeros((num_bins, ntaxa))
                beta[0] = np.dot(np.linalg.inv(diag[0]),grad[0])
                for k in range(1, num_bins):
                    beta[k]=np.dot(np.linalg.inv(diag[k]-np.matmul(low_off_diag[k-1], gamma[k-1])), grad[k]-np.dot(low_off_diag[k-1],beta[k-1]))

                mu[num_bins-1] = beta[num_bins-1]
                for k in reversed(range(num_bins-1)):
                    mu[k] = beta[k]-np.dot(gamma[k], mu[k+1])
                
                expected_states = calc_expected_states(num_bins, nsamples, dataset, mu, sigma, mixing)

                new_delta = np.zeros(ntaxa)
                for k in range(1, num_bins):
                    mu_diff = mu[k]-mu[k-1]
                    new_delta += mu_diff
                delta = new_delta/(num_bins-1)
                expected_states = calc_expected_states(num_bins, nsamples, dataset, mu, sigma, mixing)

                new_Q = np.zeros((ntaxa, ntaxa))
                for k in range(1, num_bins):
                    mu_diff = mu[k]-mu[k-1]-delta
                    mu_diff = np.reshape(mu_diff, (-1, 1))
                    new_Q += np.dot(mu_diff, mu_diff.T)
                new_Q /= (num_bins-1)

                new_Q = bound_eigenval(new_Q)

                #w, v = np.linalg.eig(new_Q)

                Q = np.copy(new_Q)
                                
                new_edge_mean = mu[0] 

                Q_edge = np.dot(new_edge_mean-edge_mean, (new_edge_mean-edge_mean).T)
                edge_mean = np.reshape(new_edge_mean, (-1, 1))
                Q_edge = bound_eigenval(Q_edge)
                #edge_mean = new_edge_mean
                expected_states = calc_expected_states(num_bins, nsamples, dataset, mu, sigma, mixing)

                #Perform the E-step      
                expected_states = calc_expected_states(num_bins, nsamples, dataset, mu, sigma, mixing)
                #print(expected_states)
                
                N_k = np.zeros(num_bins)
                for i in range(num_bins):
                    N_k[i] = np.sum(expected_states[:,i])
                mixing = N_k/nsamples
            
                #It's important to bound the mixing parameters - mixing[k] close to 0 leads to divide by 0 error
                
                for i in range(num_bins):
                    if mixing[i] < 0.001:
                        mixing[i] = 0.001
                mixing /= np.sum(mixing)
                
                expected_states = calc_expected_states(num_bins, nsamples, dataset, mu, sigma, mixing)
                
                N_k = np.zeros(num_bins)
                for i in range(num_bins):
                    N_k[i] = np.sum(expected_states[:,i])

                eps = 1e-6
                for k in range(num_bins):
                    new_sigma = np.zeros((ntaxa, ntaxa))
                    diff_big = dataset - mu[k]
                    new_sigma += np.matmul(diff_big.T*expected_states[:,k], diff_big)#np.cov(diff_big.T, ddof=0, aweights=expected_states[:,k]
                    sigma[k] = np.copy(new_sigma/(eps+N_k[k]))
                for i in range(num_bins):
                    sigma[i] = bound_eigenval(sigma[i])

                expected_states = calc_expected_states(num_bins, nsamples, dataset, mu, sigma, mixing)
                iter_l_likelihood = calc_l_likelihood(expected_states, nsamples, num_bins, dataset, mixing, mu, sigma, edge_mean, Q_edge, Q, delta)

                if np.abs((prev_l_likelihood-iter_l_likelihood)/prev_l_likelihood) < 1e-4:
                    in_row+=1
                    print("CONVERGING"+str(in_row))
                else:
                    in_row=0
                prev_l_likelihood = iter_l_likelihood
                likelihoods.append(prev_l_likelihood)
                iterations+=1
                if in_row>=5:
                    finished_flag=True
                    break

        except:
            print("Unexpected error:", sys.exc_info()[0])
            attempts+=1
            print("ATTEMPT FAILED")
            pass
        
    return mixing, sigma, delta, Q, Q_edge, edge_mean, mu, likelihoods, iterations