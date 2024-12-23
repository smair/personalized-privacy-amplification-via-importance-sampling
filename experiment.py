import fire
import os.path

from time import time

import numpy as np

import utils
import kmeans
import optimal


def main(dataset, k=25, m=0, lam=0.5, norm=2, eps=10.0, T=10, p=97.5, reps=50):
    """ A script to run an experiment and save the results to a .npz file.

    Args:
        dataset: A string of the dataset name.
        k: Number of clusters.
        m: Subset size. The setting m=-1 uses the entire data set.
        lam: The lambda parameter of the sampling distribution. The setting lam=1.0 corresponds to a uniform subsample whereas lam=0.5 corresponds to our proposed sampling strategy. Privacy-optimal sampling is achieved via lam=-1.0.
        norm: The norm for the sampling distribution. It affects the noise computation.
        eps: The target-epsilon that should be achieved.
        T: Number of k-means iterations.
        p: Percentage of data to be used. The rest is discarded as outliers. It uses norm!
        reps: Number of repetitions.

    Returns:
        It does not return anything but saves the results of that experiment in a .npz file.
    """

    # cast arguments
    k = int(k)
    m = int(m)
    lam = float(lam)
    norm = int(norm)
    eps = float(eps)
    T = int(T)
    p = float(p)
    reps = int(reps)

    config = {
        'dataset':dataset,
        'k':k,
        'm':m,
        'lam':lam,
        'norm':norm,
        'eps':eps,
        'T':T,
        'p':p,
        'reps':reps,
        'start_time':time()
    }

    # set filename of the results file
    filename = f'result_{dataset}_{k}_{m}_{lam}_{norm}_{eps}_{T}_{p}_{reps}.npz'

    print(f'Attempting to run dataset={dataset} k={k} m={m} lam={lam} norm={norm} eps={eps} T={T} p={p} reps={reps}')
    # do not run the experiment if a results file already exists!
    if os.path.exists(filename):
        print(f'ABORT: experiment {filename} already exists!')
        exit()
    else:
        # start and block the experiment
        np.savez(filename, config=config)

    # load data
    X, _ = utils.load_data(dataset)

    # preprocess data (center and remove outliers)
    X = utils.preprocess_data(X, p, norm)

    # get data information
    n, d = X.shape

    # m == -1 refers to the entire data set
    if m == -1:
        m = n

    # compute parameters and print some information
    print(f'n = {n}')
    print(f'd = {d}')
    print(f'm = {m}')
    print(f'k = {k}')
    print(f'lambda = {lam:.2f}')
    print(f'norm = {norm}')
    print(f'T = {T}')
    # assumption is that data lives in B_norm(r), get r
    r = np.linalg.norm(X, ord=norm, axis=1).max()
    print(f'r = {r:.2f}')
    if norm == 1:
        dist = np.linalg.norm(X, ord=1, axis=1)
        # x_tilde is the mean of the 1-norm of x
    elif norm == 2:
        dist = np.linalg.norm(X, ord=2, axis=1)**2
        # x_tilde is the mean of the squared 2-norm of x
    else:
        raise NotImplementedError('Only norm=1 and norm=2 are implemented.')
    x_tilde = np.mean(dist)
    print(f'x_tilde = {x_tilde:.2f}')

    # setup for the experiment
    res_C = []
    res_obj = []
    res_obj_sub = []
    res_time = []
    res_time_weights = []
    res_time_sample = []
    res_target_epsilon = [] # only used for privacy-optimal

    beta_sum = None
    beta_count = None
    q = None
    weights = None
    target_epsilon = None

    t_weight1 = None
    t_weight2 = None

    # get sampling probabilities and weights
    if lam==1.0:
        # uniform sampling
        print('computing uniform weights')

        # track the time of the weight computation (should only matter for privacy-optimal weights)
        t_weight1 = time()

        q = m/n * np.ones(n)
        weights = 1.0/q

        # end of time tracking
        t_weight2 = time()
        res_time_weights.append(t_weight2-t_weight1)

    elif lam==0.5:
        # coreset-inspired sampling
        print('computing coreset-inspired weights')

        # track the time of the weight computation (should only matter for privacy-optimal weights)
        t_weight1 = time()

        # re-compute the distances to also track the time of computing those
        if norm == 1:
            dist = np.linalg.norm(X, ord=1, axis=1)
        elif norm == 2:
            dist = np.linalg.norm(X, ord=2, axis=1)**2
        else:
            raise NotImplementedError('Only norm=1 and norm=2 are implemented.')

        q = lam*m/n + (1-lam)*m*dist/(n*x_tilde)
        weights = 1.0/q

        # end of time tracking
        t_weight2 = time()
        res_time_weights.append(t_weight2-t_weight1)

    else:
        # negative lambda means privacy-optimal weights
        print('computing privacy-optimal weights')

        X_lp = np.linalg.norm(X, ord=norm, axis=1)

        # find B for target m
        B = optimal.find_B_for_target_m(eps, d, m, norm, r, T, X_lp)

        # re-compute noise scales
        beta_sum, beta_count = utils.compute_noise_scales(B, d, r, T)

        # track the time of the weight computation (should only matter for privacy-optimal weights)
        t_weight1 = time()

        # compute privacy-optimal weights
        weights = optimal.pais_rs.all_optimal_weights(X_lp, beta_count, beta_sum, T, eps)
        weights = np.array(weights)
        q = 1/weights
        # m_opt = np.sum(1/weights)

        # end of time tracking
        t_weight2 = time()
        res_time_weights.append(t_weight2-t_weight1)


    print(f'computing the weights took {t_weight2-t_weight1} seconds')


    # compute noise scales for unif and core
    if lam >= 0.0:
        B = utils.find_B_for_target_epsilon_on_subset_new(d, lam, m, n, norm, r, T, eps, x_tilde)
        print(f'B = {B:.6f}')

        beta_sum, beta_count = utils.compute_noise_scales(B, d, r, T)
        print(f'beta_sum = {beta_sum:.2f}')
        print(f'beta_count = {beta_count:.2f}')


    # repeat the experiment several times
    for rep in range(reps):
        print(f'repetition {rep+1}/{reps}')
        # set seed
        np.random.seed(rep)

        # track the time of the subset computation
        t_sample1 = time()
        # subset generation
        S = None
        _weights = None
        if m == n:
            # m==-1 (m=n) means full data set
            S = X
            # compute weights
            _weights = np.ones(n)
        else:
            # we now sample m>0 points
            sample_mask = np.random.rand(n) <= q
            print(f'\tsampled {np.sum(sample_mask)} points (m={m}, n={n})')
            S = X[sample_mask]
            # compute weights
            _weights = weights[sample_mask]
        # end of time tracking
        t_sample2 = time()

        # initialize the centers uniformly
        ind = np.random.choice(S.shape[0], k, replace=False)
        C = S[ind]

        # run DP-Lloyd
        C_hist, obj_hist_sub, t_hist = kmeans.DP_k_means(S, C, beta_sum, beta_count, w=_weights, norm=norm, max_it=T)

        print('recompute objective on all data')
        obj_hist = kmeans.obj(X, C_hist)

        # remember the results
        res_C.append(C_hist)
        res_obj.append(obj_hist)
        res_obj_sub.append(obj_hist_sub)
        res_time_sample.append(t_sample2-t_sample1)
        res_time.append(t_hist)
        res_target_epsilon.append(target_epsilon)

        # save and overwrite results
        np.savez(filename,
                 config=config,
                 beta_sum=beta_sum,       # noise scale
                 beta_count=beta_count,   # noise scale
                 weights=weights,         # data weights
                 q=q,                     # sampling distribution
                 rep=rep,
                 res_C=res_C,
                 res_obj=res_obj,
                 res_obj_sub=res_obj_sub,
                 res_time=res_time,
                 res_time_weights=res_time_weights,
                 res_time_sample=res_time_sample,
                 res_target_epsilon=res_target_epsilon)


if __name__ == '__main__':
  fire.Fire(main)


