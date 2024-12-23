import numpy as np
from time import time

# constant for DP-k-means
rho = 0.225


def obj(X, C_hist):
    """
    Computes the k-means objective on data X using a set of centers C.

    Args:
        X: A data matrix
        C_hist: A (historical) list of centers C (over, e.g., T iterations)

    Returns:
        A list of objective function values.
    """
    n = X.shape[0]
    T = C_hist.shape[0]
    return np.array([np.sum([np.sum((X[i]-C_hist[it])**2, axis=1).min()
                             for i in range(n)])
                     for it in range(T)])


def k_means(X, C, w=None, max_it=30):
    """ Implementation of k-means.

    Args:
        X: A data matrix
        C: Initial cluster centers
        w: An array of weights (per data point)
        max_it: Desired number of iterations

    Returns:
        C_hist: A list of centers C over iterations
        obj_hist: A list of objective function values over iterations
        t_hist: A list of computation times over iterations
    """
    # number of data points and their dimensionality
    n, d = X.shape
    # number of clusters
    k = C.shape[0]
    # set weight vector and fix shape
    if w is None:
        w = np.ones(n)
    w = w.reshape(-1,1)
    Xw = np.multiply(w, X)
    # keep track of the cluster centers, objective function value, and time
    C_hist = [C]
    obj_hist = []
    t_hist = [time()]
    # compute the first assignment, i.e., closest cluster index per point
    assignment = np.array([np.sum((X[i]-C)**2, axis=1).argmin()
                           for i in range(n)])
    # compute the objective function value
    obj_hist.append(np.sum(np.multiply(w, (X-C[assignment])**2)))
    # conduct some iterations
    for it in range(max_it):
        # compute the cluster sizes
        cluster_size = [np.sum(w[assignment==j]) for j in range(k)]
        # update the centers
        C = np.array([(np.sum(Xw[assignment==j], axis=0)) / cluster_size[j]
                      for j in range(k)])
        # compute new assignment
        assignment = np.array([np.sum((X[i]-C)**2, axis=1).argmin()
                               for i in range(n)])
        # compute the objective function value
        obj_hist.append(np.sum(np.multiply(w, (X-C[assignment])**2)))
        # keep track of the cluster centers and time
        C_hist.append(C)
        t_hist.append(time())
    # note that the length of those histories is max_it+1
    C_hist = np.array(C_hist)
    obj_hist = np.array(obj_hist)
    t_hist = np.array(t_hist) - t_hist[0]
    return C_hist, obj_hist, t_hist


def DP_k_means(X, C, beta_sum, beta_count, w=None, norm=1, max_it=30):
    """ Implementation of DP k-means according to Su et al. (2016).

    Args:
        X: A data matrix
        C: Initial cluster centers
        beta_sum: The scale parameter for the sum noise.
        beta_count: The scale parameter for the count noise.
        w: An array of weights (per data point)
        norm: The norm for the sampling distribution. It affects the noise computation.
        max_it: Desired number of iterations

    Returns:
        C_hist: A list of centers C over iterations
        obj_hist: A list of objective function values over iterations
        t_hist: A list of computation times over iterations
    """
    # number of data points and their dimensionality
    n, d = X.shape
    # number of clusters
    k = C.shape[0]
    # set weight vector and fix shape
    if w is None:
        w = np.ones(n)
    w = w.reshape(-1,1)
    Xw = np.multiply(w, X)
    # keep track of the cluster centers, objective function value, and time
    C_hist = [C]
    obj_hist = []
    t_hist = [time()]
    # compute the first assignment, i.e., closest cluster index per point
    assignment = np.array([np.sum((X[i]-C)**2, axis=1).argmin()
                           for i in range(n)])
    # compute the objective function value
    obj_hist.append(np.sum(np.multiply(w, (X-C[assignment])**2)))
    # conduct some iterations
    for it in range(max_it):
        # compute the noise for the count
        count_noise = np.random.laplace(loc=0.0, scale=beta_count, size=k)
        # compute the noisy cluster sizes
        noisy_cluster_size = [np.sum(w[assignment==j]) + count_noise[j]
                              for j in range(k)]
        # compute the noise for the sum
        randn = np.random.randn(k, d)
        zeta = randn / np.linalg.norm(randn, ord=norm, axis=1).reshape(-1, 1)
        nu = np.random.exponential(beta_sum, k)
        sum_noise = zeta*nu.reshape(-1, 1)
        # update the centers
        C = np.array([(np.sum(Xw[assignment==j], axis=0) + sum_noise[j]) / noisy_cluster_size[j]
                      for j in range(k)])
        # compute new assignment
        assignment = np.array([np.sum((X[i]-C)**2, axis=1).argmin()
                               for i in range(n)])
        # compute the objective function value
        obj_hist.append(np.sum(np.multiply(w, (X-C[assignment])**2)))
        # keep track of the cluster centers and time
        C_hist.append(C)
        t_hist.append(time())
    # note that the length of those histories is max_it+1
    C_hist = np.array(C_hist)
    obj_hist = np.array(obj_hist)
    t_hist = np.array(t_hist) - t_hist[0]
    return C_hist, obj_hist, t_hist



