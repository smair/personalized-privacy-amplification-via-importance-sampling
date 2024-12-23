import re
import os.path
import numpy as np
import pandas as pd
import scipy
import scipy.optimize
import scipy.special
from tqdm import tqdm
from sklearn.datasets import load_svmlight_file, fetch_california_housing
from scipy.io import loadmat

import kmeans

# specify the location of where the data is stored
data_path = '/home/sebastian/data/'


def logexpm1(x):
    """Compute log(exp(x) - 1) in a numerically stable way."""
    return scipy.special.logsumexp([x, complex(0.0, np.pi)]).real


def logtheta(u, v):
    """Compute log(v * (exp(u/v) - 1)) in a numerically stable way."""
    return np.log(v)+logexpm1(u/v)


def compute_noise_scales(B, d, r, T):
    beta_sum = np.sqrt(T*r/B) * np.cbrt(d/(2*kmeans.rho))
    beta_count = beta_sum * np.cbrt(4*d*kmeans.rho**2)

    return beta_sum, beta_count


def compute_subset_epsilon_new(B, d, lam, m, n, norm, r, T, x_tilde, grid_num=100):
    """Compute the DP parameter of coreset-subsampled k-means for an arbitrary norm.
    Uses grid search over a linearly spaced grid of `grid_num` points."""
    beta_sum, beta_count = compute_noise_scales(B, d, r, T)
    def epsilon(z):
        a_plus_tz = T / beta_count + T / beta_sum * z
        b_plus_sz2 = lam * m / n + (1 - lam) * m / n / x_tilde * z ** norm
        logtheta_ = logtheta(a_plus_tz, b_plus_sz2)
        return scipy.special.logsumexp([0, logtheta_])
    z_grid = np.linspace(0, r, num=grid_num)
    epsilons = [epsilon(z) for z in z_grid]
    return max(epsilons)


def compute_subset_epsilon(B, d, lam, m, n, norm, r, T, x_tilde):
    beta_sum, beta_count = compute_noise_scales(B, d, r, T)

    A1 = (1/beta_count + r/beta_sum)*T
    A1_tick = T/beta_count
    A2 = m/n*(lam+(1-lam)*(r**norm)/x_tilde)
    A2_tick = lam*m/n

    """
    theta(u,v) = (exp(u/v)-1)*v
    log(theta(u,v)) = log(exp(u/v)-1)+log(v)
    argmax(theta_1, theta_2) = argmax(log(theta_1), log(theta_2))
    eps = log(1+max(theta1, theta2)) = log(exp(log(theta_1))+exp(log(theta_2)))
    """

    max_ = np.max([logtheta(A1, A2), logtheta(A1_tick, A2_tick)])
    eps = np.logaddexp(0, max_)

    return eps


def find_B_for_target_epsilon_on_subset_new(d, lam, m, n, norm, r, T, target_eps, x_tilde, lower=1e-7, upper=5e4):
    def f(B):
        return compute_subset_epsilon_new(B,d,lam,m,n,norm,r,T,x_tilde)-target_eps

    B = scipy.optimize.bisect(f, lower, upper)

    return B


def find_B_for_target_epsilon_on_subset(d, lam, m, n, norm, r, T, target_eps, x_tilde, lower=1e-7, upper=5e4):
    def f(B):
        return compute_subset_epsilon(B,d,lam,m,n,norm,r,T,x_tilde)-target_eps

    B = scipy.optimize.bisect(f, lower, upper)

    return B


def preprocess_data(X, p, norm):
    # get data information
    n, d = X.shape

    # assumption: data is centered
    X = X - X.mean(0)
    assert(np.allclose(X.mean(0), np.zeros(d)))

    # if not all data shall be used
    if p < 100.0:
        # throw away p percentile points with the highest norms
        norm_per_datapoint = np.linalg.norm(X, ord=norm, axis=1)
        mask = norm_per_datapoint < np.percentile(norm_per_datapoint, [p]).item()
        X = X[mask]
        # print(f'reduced data from n={n} to n={np.sum(mask)} data points, p={p}, np/100={n*p/100}')

        # assumption: data is centered
        X = X - X.mean(0)
        assert(np.allclose(X.mean(0), np.zeros(d)))

    return X


def load_data(dataset):
    X = []
    y = []

    if dataset == 'covertype':  # (581012, 54)
        # Forest cover type
        # https://archive.ics.uci.edu/ml/datasets/covertype
        X, y = load_svmlight_file(data_path + 'covtype.libsvm.binary')
        X = np.asarray(X.todense())
    elif dataset == 'ijcnn1':  # (49990, 22)
        # https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
        X, y = load_svmlight_file(data_path + "ijcnn1/ijcnn1")
        X = np.asarray(X.todense())
    elif dataset == 'song':  # (515345, 90)
        # YearPredictionMSD is a subset of the Million Song Dataset
        # https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd
        data = np.loadtxt(
            data_path + 'YearPredictionMSD.txt', skiprows=0, delimiter=','
        )
        X = data[:, 1:]
        y = data[:, 0]
    elif dataset == 'pose':  # (35832, 48)
        # ECCV 2018 PoseTrack Challenge
        # http://vision.imar.ro/human3.6m/challenge_open.php
        X = []
        cache_file = data_path + 'Human3.6M/ECCV18_Challenge/train_cache.npz'
        if os.path.exists(cache_file):
            print('utils.load_data(): loading cache file for pose data')
            npz = np.load(cache_file)
            X = npz['X']
        else:
            X = []
            for i in tqdm(range(1, 35832 + 1), desc='loading pose'):
                f = data_path + 'Human3.6M/ECCV18_Challenge/Train/POSE/{:05d}.csv'.format(i)
                data = np.loadtxt(f, skiprows=0, delimiter=",")
                X.append(data[1:, :].flatten())
            X = np.array(X)
            print('utils.load_data(): saving cache file for pose data')
            np.savez(cache_file, X=X)
    elif dataset == 'kdd-protein': # (145751, 74)
        # KDD Cup 2004
        # http://osmot.cs.cornell.edu/kddcup/datasets.html
        #
        # Protein Homology Dataset
        #
        # Example:
        # 279 261532 0 52.00 32.69 ... -0.350 0.26 0.76
        #
        # 279 is the BLOCK ID.
        # 261532 is the EXAMPLE ID.
        # The "0" in the third column is the target value. This indicates that this
        #      protein is not homologous to the native sequence (it is a decoy).
        #      If this protein was homologous the target would be "1".
        # Columns 4-77 are the input attributes.

        data = pd.read_csv(data_path+'KDD_protein/bio_train.dat', sep='\t', skiprows=0, header=None)
        X = np.asarray(data)[:,3:]
    elif dataset == 'rna': # (488565, 8)
        # https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
        X_train, y_train = load_svmlight_file(data_path+'RNA/libsvmtools_datasets/cod-rna')
        X_train = np.asarray(X_train.todense())
        X_val, y_val = load_svmlight_file(data_path+'RNA/libsvmtools_datasets/cod-rna.t')
        X_val = np.asarray(X_val.todense())
        X_rest, y_rest = load_svmlight_file(data_path+'RNA/libsvmtools_datasets/cod-rna.r')
        X_rest = np.asarray(X_rest.todense())

        X = np.vstack((X_train,X_val,X_rest))
        y = np.hstack((y_train,y_val,y_rest))
    elif dataset == 'miniboone': # (130064, 50)
        # https://archive.ics.uci.edu/ml/datasets/MiniBooNE+particle+identification
        data = pd.read_csv(data_path+'MiniBooNE_PID.txt', sep='\s\s*', skiprows=[0], header=None, engine='python')
        X = np.asarray(data)
    elif dataset == 'fma':
        # https://github.com/mdeff/fma
        data = pd.read_csv(data_path+'fma_metadata/features.csv', index_col=0, header=[0, 1, 2])
        X = np.asarray(data)
    else:
        raise NotImplementedError

    return X, y
