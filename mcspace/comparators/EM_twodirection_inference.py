import numpy as np
from scipy.stats import multivariate_normal, invwishart
import pickle as pkl
from skbio.stats.composition import ilr
from sklearn.metrics.cluster import v_measure_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import copy
import multiprocessing as mp
import time
import random
from .block_pentadiagonal import solve_pent_diagonal_thomas
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
        for i in range(int(num_bins/2)):
            dist+=np.linalg.norm(mu_perm[2*i+1]-mu_perm[2*i])
            if i<int(num_bins/2)-1:
                dist+=np.linalg.norm(mu_perm[2*i+3]-mu_perm[2*i+1])
                dist+=np.linalg.norm(mu_perm[2*i+2]-mu_perm[2*i])
        return dist
    perms = list(itertools.permutations([i for i in range(len(mu))]))
    best_dist=None
    best_mu = copy.deepcopy(mu)
    new_mu = copy.deepcopy(mu)
    num_times=0
    best_perm = None
    for i in range(len(perms)):
        for j in range(len(mu)):
            new_mu[j] = mu[perms[i][j]]
        if best_dist is None or dist(new_mu)<best_dist:
            num_times+=1
            best_dist = dist(new_mu)
            best_mu = copy.deepcopy(new_mu)
            best_perm = perms[i]
    return best_mu, best_perm
def rmse(predictions, targets):
    """
    For 2D array: take the rmse for each row, get the average of those. 
    """
    error = 0
    for i in range(len(predictions)):
        error += np.sqrt(((predictions[i] - targets[i]) ** 2).mean())/np.mean(np.abs(targets[i]))
    error /= len(predictions)
    return error

def initialize_params(dataset, K, num_dim):
    """
    Parameters
    ----------
    K: number of bins in longer direction
        We construct 2K bins
    num_dim : Int
        Number of effective taxa (ntaxa-1 for philr)
    Returns
    -------
    Initialized params for mixing param,noise covariances, spatial covariance, 
    edge mean, and edge covariance, and all bin means
    
    Note: I also show alternative initialization methods in this code
    """
    num_bins = 2*K
    mixing = np.zeros(num_bins)
    sigma = np.zeros((num_bins, num_dim, num_dim))
    for i in range(num_bins):
        mixing[i] = 1/num_bins
        sigma[i] = 0.1*np.eye(num_dim)
    iw = invwishart(df=num_dim+1, scale=np.identity(num_dim))
    
    sigma = iw.rvs((num_bins,))
    for i in range(num_bins):
        sigma[i] = set_trace(sigma[i], 0.1)
    Q_edge = iw.rvs((1,))
    Q_edge = set_trace(Q_edge, 0.1)
    
    Q = iw.rvs((1,))
    Q = set_trace(Q, 0.1)

    #One option for delta init: Use pca two largest components
    pca = PCA(n_components=2)
    pca.fit(dataset)
    delta = pca.components_[0]*pca.singular_values_[0]
    delta_perp = pca.components_[1]*pca.singular_values_[1]

    #Simulate the rest of the means with Kmeans
    mu = np.zeros((num_bins, num_dim))


    
    basic_kmeans = KMeans(n_clusters=num_bins, n_init=10).fit(dataset) #We DO NOT set the random state- want a different ordering everytime
    mu = basic_kmeans.cluster_centers_
    mu = rearrange_means(mu)
    
    #What the algo actually uses: initialize with the naive GMM
    fitted = basic_EM_algo(dataset, num_bins, threshold=1e-3)
    basic_mixing, basic_sigma, basic_mu, likelihoods, iterations = fitted
    mu = copy.deepcopy(basic_mu)
    mu, perm = rearrange_means(mu)
    for i in range(len(perm)):
        sigma[i] = basic_sigma[perm[i]]
        mixing[i] = basic_mixing[perm[i]]
    delta_perp = np.zeros(num_dim)
    delta = np.zeros(num_dim)
    for i in range(K):
        delta_perp+=(mu[2*i+1]-mu[2*i])/K
        if i!=K-1:
            delta+=(mu[2*i+2]-mu[2*i])/(2*K-2)
            delta+=(mu[2*i+3]-mu[2*i+1])/(2*K-2)
    edge_mean = mu[0]
    
    
    #A small snippet of code to show how you can assign samples to bins
    Z = np.zeros((dataset.shape[0],2*K))
    for n in range(dataset.shape[0]):
        for k in range(2*K):
            Z[n,k]=mixing[k]*multivariate_normal.pdf(dataset[n], mu[k], sigma[k])
    best_bins = np.argmax(Z,axis=1)
    #print(best_bins)

    return mixing, sigma, delta, delta_perp, Q, Q_edge, edge_mean, mu

def calc_l_likelihood(expected_states, nsamples, K, dataset, mixing, mu, sigma, edge_mu, edge_Q, Q, delta, delta_perp):
    num_bins = 2*K
    log_likelihood = 0
    
    for k in range(num_bins):
        log_likelihood += np.sum(expected_states[:,k]*multivariate_normal.logpdf(dataset, mu[k], sigma[k]))
        log_likelihood += np.log(mixing[k])*np.sum(expected_states[:,k])

    log_likelihood += multivariate_normal.logpdf(mu[0], edge_mu.flatten(), edge_Q)

    log_likelihood += multivariate_normal.logpdf(mu[1], mu[0]+delta_perp, Q)
    for k in range(1, K):
        log_likelihood += multivariate_normal.logpdf(mu[2*k], mu[2*(k-1)]+delta, Q) 
        log_likelihood += multivariate_normal.logpdf(mu[2*k+1], 0.5*(mu[2*k]+mu[2*k-1]+delta+delta_perp), Q)
    return log_likelihood

def calc_expected_states(K, nsamples, dataset, mu, sigma, mixing):
    num_bins = 2*K
    probabilities = np.zeros((num_bins, nsamples)) #log probability
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
    A : some symmetric matrix
    Returns
    -------
    recalculated A

    """
    w, v = np.linalg.eigh(A)
    #print(A)
    #print(w)
    eig_mat = np.zeros((len(w), len(w)))
    for i in range(len(w)):
        if w[i] < 1e-3:
            w[i]=1e-3
        if isinstance(w[i], complex):
            w[i]=1e-3
        eig_mat[i,i]=w[i]
    #eig mat is now the new diagonal matrix
    return v.dot(eig_mat).dot(v.T)
def EM_algo(dataset, K):
    """
    Parameters
    ----------
    dataset : 2D np array
        number of samples x number of taxa

    Returns
    -------
    Probably should return the bin means and the expected labels or something
    """
    def E_Sigma(expectation_states, sigma, k):
        sig_inv = np.linalg.inv(sigma[k])
        return np.sum(expectation_states[:,k])*sig_inv
    num_bins = 2*K
    ntaxa = dataset.shape[1]
    nsamples = dataset.shape[0]
    finished_flag = False
    #Just initializing them so it's easy to return them outside of the loop
    mixing, sigma, delta, delta_perp, Q, Q_edge, edge_mean, mu = [None for i in range(8)]
    
    likelihoods = []
    iterations = 0
    attempts = 0
    
    while finished_flag is False:
        print("ATTEMPT")
        try:
            likelihoods = []
            iterations = 0
            mixing, sigma, delta, delta_perp, Q, Q_edge, edge_mean, mu = initialize_params(dataset, K, ntaxa)
            edge_mean = np.reshape(edge_mean, (-1, 1))
            prev_l_likelihood=-0.01
            expected_states = calc_expected_states(K, nsamples, dataset, mu, sigma, mixing)
            
            in_row = 0 #counts how many times in a row the likelihood has changed less than threshold

            for max_iterations in range(500): #500 is just an arbitary number of max iterations (rarely runs that long)

                expected_states = calc_expected_states(K, nsamples, dataset, mu, sigma, mixing)
                
                new_delta_perp = -(K-1)*delta/(K+3)+  4/(K+3)*(0.75*mu[1]-mu[0]+0.5*mu[-1])
                for k in range(1, K-1):
                    new_delta_perp += 1/(K+3)*mu[2*k+1]
                for k in range(1, K):
                    new_delta_perp -= 1/(K+3)*mu[2*k]
                delta_perp = new_delta_perp

                #Solving a system of equations to update Delta
                coefficient_matrix = np.zeros((len(delta)+1, len(delta)+1))
                intercept = np.zeros(len(delta))
                for k in range(1,K):
                    intercept += mu[2*k]+0.5*mu[2*k+1]-mu[2*k-2]-0.25*mu[2*k-1]-0.25*mu[2*k]
                    
                intercept += -0.25*(K-1)*delta_perp
                intercept /= (1.25*(K-1))
                intercept = np.append(intercept, 0) #the last element is a 0 (represents the orthogonality constraint)

                lambda_coefficient = -1/(1.25*(K-1))*np.matmul(Q, delta_perp)
                for i in range(len(delta)):
                    coefficient_matrix[i,i]=1
                    coefficient_matrix[i,-1]=lambda_coefficient[i]
                    coefficient_matrix[-1, i]=delta_perp[i]
                answer = np.matmul(np.linalg.inv(coefficient_matrix), intercept)

                delta = answer[:-1]
                
                expected_states = calc_expected_states(K, nsamples, dataset, mu, sigma, mixing)
                
                # Now we update the means using Thomas Algorithm
                Q_inv = np.linalg.inv(Q)
                
                diag = np.zeros((num_bins, ntaxa, ntaxa))
                diag[0] = -np.linalg.inv(Q_edge)-2*Q_inv-E_Sigma(expected_states, sigma, 0)
                diag[1] = -1.25*Q_inv-E_Sigma(expected_states, sigma,1)
                for i in range(1, K-1):
                    diag[2*i] = -2.25*Q_inv-E_Sigma(expected_states, sigma,2*i)
                    diag[2*i+1] = -1.25*Q_inv-E_Sigma(expected_states, sigma,2*i+1)
                diag[-2] = -1.25*Q_inv-E_Sigma(expected_states, sigma, 2*K-2)
                diag[-1] = -Q_inv-E_Sigma(expected_states, sigma, 2*K-1)
                off_diag = []
                off_off_diag = []
 
                for i in range(num_bins): #note that num_bins is even
                    if i==0:
                        off_diag.append(Q_inv)
                        off_off_diag.append(Q_inv)
                    elif i==1:
                        off_diag.append(-0.25*Q_inv)
                        off_off_diag.append(0.5*Q_inv)
                    elif i%2==0:
                        if i<(num_bins-2):
                            off_diag.append(0.5*Q_inv)
                            off_off_diag.append(Q_inv)
                        if i==num_bins-2: #Second to last diagonal element doesn't have off-off-diagonal element
                            off_diag.append(0.5*Q_inv)
                    elif i%2==1:
                        if i<(num_bins-2):
                            off_diag.append(-0.25*Q_inv)
                            off_off_diag.append(0.5*Q_inv)
                        #if i==num_bins-1, then there are no off_diag elements 
                grad = np.zeros((num_bins, ntaxa))
                for k in range(num_bins):
                    sig_inv = np.linalg.inv(sigma[k])
                    for n in range(nsamples):
                         grad[k]+=(expected_states[n,k]*np.dot(sig_inv, dataset[n]))
                    if k>=2:
                        if k%2==0:
                            if k<num_bins-2 and k>0:
                                grad[k]+=np.dot(Q_inv, -0.25*(delta+delta_perp)).flatten()
                            if k==num_bins-2:
                                grad[k]+=np.dot(Q_inv, 0.75*delta-0.25*delta_perp).flatten()
                        else:
                            if k<num_bins-2:
                                grad[k]+=np.dot(Q_inv, 0.25*(delta+delta_perp)).flatten()
                            if k==num_bins-1:
                                grad[k]+=np.dot(Q_inv, 0.5*(delta+delta_perp)).flatten()
                grad[0]+=np.dot(np.linalg.inv(Q_edge), edge_mean).flatten()+np.dot(Q_inv, -1*(delta_perp+delta)).flatten()
                grad[1]+=np.dot(Q_inv, -0.25*delta+0.75*delta_perp).flatten()
                
                mu = solve_pent_diagonal_thomas(off_off_diag, off_diag, diag, grad)

                expected_states = calc_expected_states(K, nsamples, dataset, mu, sigma, mixing)
                
                new_Q = np.zeros((ntaxa, ntaxa))
                for k in range(1, K):
                    mu_diff = mu[2*k]-mu[2*(k-1)]-delta
                    mu_diff = np.reshape(mu_diff, (-1, 1))
                    new_Q += np.dot(mu_diff, mu_diff.T)
                    mu_diff = mu[2*k+1]-0.5*(mu[2*(k-1)+1]+mu[2*k])-0.5*(delta+delta_perp)
                    new_Q += np.dot(mu_diff, mu_diff.T)
                mu_diff = mu[2]-mu[1]-delta_perp
                mu_diff = np.reshape(mu_diff, (-1, 1))
                new_Q += np.dot(mu_diff, mu_diff.T)
                new_Q /= (2*K-1)
                new_Q = bound_eigenval(new_Q)
                Q = np.copy(new_Q)
                new_edge_mean = mu[0] 
    
                Q_edge = np.dot(new_edge_mean-edge_mean, (new_edge_mean-edge_mean).T)
                edge_mean = np.reshape(new_edge_mean, (-1, 1))
                Q_edge = bound_eigenval(Q_edge)
                
                expected_states = calc_expected_states(K, nsamples, dataset, mu, sigma, mixing)
                
                N_k = np.zeros(num_bins)
                for i in range(num_bins):
                    N_k[i] = np.sum(expected_states[:,i])
                mixing = N_k/nsamples
            
                #It's important to bound the mixing parameters - mixing[k] close to 0 leads to divide by 0 error
                
                for i in range(num_bins):
                    if mixing[i] < 0.001:
                        mixing[i] = 0.001
                mixing /= np.sum(mixing)
                
                
                expected_states = calc_expected_states(K, nsamples, dataset, mu, sigma, mixing)
                                
                N_k = np.zeros(num_bins)
                for i in range(num_bins):
                    N_k[i] = np.sum(expected_states[:,i])
                
                eps = 1e-6
                for k in range(num_bins):
                    new_sigma = np.zeros((ntaxa, ntaxa))
                    diff_big = dataset - mu[k]
                    new_sigma += np.matmul(diff_big.T*expected_states[:,k], diff_big)
                    sigma[k] = np.copy(new_sigma/(eps+N_k[k]))
                for i in range(num_bins):
                    sigma[i] = bound_eigenval(sigma[i])

                expected_states = calc_expected_states(K, nsamples, dataset, mu, sigma, mixing)
                
                iter_l_likelihood = calc_l_likelihood(expected_states, nsamples, K, dataset, mixing, mu, sigma, edge_mean, Q_edge, Q, delta, delta_perp)

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
    
    return mixing, sigma, delta, delta_perp, Q, Q_edge, edge_mean, mu, likelihoods, iterations