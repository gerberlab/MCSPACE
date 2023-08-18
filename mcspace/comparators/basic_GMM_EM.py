import numpy as np
from scipy.stats import multivariate_normal, invwishart
from sklearn.cluster import KMeans

def initialize_params(dataset, num_bins, num_dim):
    """
    Parameters
    ----------
    num_bins : Int
        Number of Bins
    num_dim : Int
        Number of effective taxa (ntaxa-1 for philr)
    Returns
    -------
    Initialized params for mixing param,noise covariances, and all bin means
    """
    mixing = np.zeros(num_bins)
    sigma = np.zeros((num_bins, num_dim, num_dim))
    for i in range(num_bins):
        mixing[i] = 1/num_bins
        sigma[i] = 0.1*np.eye(num_dim)

    iw = invwishart(df=num_dim+1, scale=np.identity(num_dim))
    sigma = 0.1*iw.rvs((num_bins,))

    #Now simulate the rest of the means
    mu = np.zeros((num_bins, num_dim))
    basic_kmeans = KMeans(n_clusters=num_bins).fit(dataset) #We DO NOT set the random state- want a different ordering everytime
    mu = basic_kmeans.cluster_centers_
    
    return mixing, sigma, mu

def calc_l_likelihood(expected_states, nsamples, num_bins, dataset, mixing, mu, sigma):
    log_likelihood = 0
    for k in range(num_bins):
        log_likelihood += np.sum(expected_states[:,k]*multivariate_normal.logpdf(dataset, mu[k], sigma[k]))
        log_likelihood += np.log(mixing[k])*np.sum(expected_states[:,k])
    return log_likelihood

def calc_expected_states(num_bins, nsamples, dataset, mu, sigma, mixing):
    #Performing the E-step in the EM algorithm
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

def EM_algo(dataset, num_bins, threshold=1e-4):
    """
    Parameters
    ----------
    dataset : 2D np array
        number of samples x number of taxa
    num_bins: int
        Number of mixtures to learn
    threshold: float
        Convergence criteria - need relative increase less than threshold for 5 consecutive iterations
    Returns
    -------
    Returns model params, likelihood from each step, and number of iterations requireds
    """

    ntaxa = dataset.shape[1]
    nsamples = dataset.shape[0]
    finished_flag = False
    
    #Just initializing them so it's easy to return them outside of the loop
    mixing, sigma, mu = initialize_params(dataset, num_bins, ntaxa)

    likelihoods = []
    iterations = 0
    attempts = 0
    while finished_flag is False:
        print("ATTEMPT")
        try:
            likelihoods = []
            iterations = 0
            mixing, sigma, mu = initialize_params(dataset, num_bins, ntaxa)

            prev_l_likelihood=-0.01
            expected_states = calc_expected_states(num_bins, nsamples, dataset, mu, sigma, mixing)
            
            in_row = 0 #counts how many times in a row the likelihood has changed less than the convergence criteria
            for z in range(50000): 
                expected_states = calc_expected_states(num_bins, nsamples, dataset, mu, sigma, mixing)
                N_k = np.zeros(num_bins)
                for i in range(num_bins):
                    N_k[i] = np.sum(expected_states[:,i])
                for k in range(num_bins):
                    mu[k] = 0
                    for n in range(nsamples):
                        mu[k]+= (expected_states[n,k]*dataset[n])/N_k[k]

                expected_states = calc_expected_states(num_bins, nsamples, dataset, mu, sigma, mixing)

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
                    new_sigma += np.matmul(diff_big.T*expected_states[:,k], diff_big)
                    sigma[k] = np.copy(new_sigma/(eps+N_k[k]))
                for i in range(num_bins):
                    sigma[i] = bound_eigenval(sigma[i])
                
                expected_states = calc_expected_states(num_bins, nsamples, dataset, mu, sigma, mixing)
                iter_l_likelihood = calc_l_likelihood(expected_states, nsamples, num_bins, dataset, mixing, mu, sigma)
                                
                if np.abs((prev_l_likelihood-iter_l_likelihood)/prev_l_likelihood) < threshold:
                    in_row+=1
                else:
                    in_row=0
                prev_l_likelihood = iter_l_likelihood
                likelihoods.append(prev_l_likelihood)
                iterations+=1
                if in_row>=5:
                    finished_flag=True
                    break
        except:
            attempts+=1
            print("ATTEMPT FAILED")
            pass
    return mixing, sigma, mu, likelihoods, iterations
