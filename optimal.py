import itertools as it
import logging
from typing import Callable

import numpy as np
from scipy import optimize as sp_optimize
from scipy import special

import pais_rs
import utils


def logexpm1(x):
    """Compute log(exp(x) - 1) in a numerically stable way.

    Args:
        x: Scalar or array-like. Anything that np.asarray can convert to a numpy array.

    Returns:
        Scalar or array with the same shape as x.
    """
    x = np.asarray(x)
    zeros = np.zeros_like(x)
    b1 = np.ones_like(x)
    b2 = -np.ones_like(zeros)
    # Implement subtraction via the b parameter (see documentation of logsumexp)
    return special.logsumexp([x, zeros], b=[b1, b2], axis=0)


def amplify(pdp_epsilon: float, weight: float) -> float:
    """Amplify a privacy profile by a given weight.

    Args:
        pdp_epsilon: The PDP profile evaluated at weight.
        weight: The importance weight used for sampling.
        xtol: The tolerance in terms of absolute error.

    Returns:
        The subsampled PDP.
    """
    if weight < 1:
        raise ValueError("weight must be greater than or equal to 1.")
    if pdp_epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    logexpm1_ = logexpm1(pdp_epsilon)
    log_second_term = logexpm1_ - np.log(weight)
    return np.logaddexp(log_second_term, 0.)


def _upper_bracket_brute_force(privacy_profile: Callable[[float], float],
                               target_epsilon: float) -> float:
    """Find a w such that privacy_profile(w) >= target_epsilon. May take up to log(w') evaluations of privacy_profile where w' is the largest w such that privacy_profile(w) = target_epsilon. If such a w does not exist, the function will not terminate.

    Args:
        privacy_profile: A function that takes an importance weight and returns the privacy loss.
        target_epsilon: The epsilon parameter of the target privacy guarantee. Must be greater than or equal to privacy_profile(1)

    Returns:
        The upper bracket for the given privacy profile.
    """
    def sampled_privacy(w):
        return logexpm1(privacy_profile(w)) - np.log(w) - logexpm1(target_epsilon)

    candidates = (2**i for i in it.count())
    for w in candidates:
        if sampled_privacy(w) >= 0:
            return float(w)


def _upper_bracket_analytical(epsilon_1: float, diff_1: float, target_epsilon: float, strong_convexity: float) -> float:
    """Compute an analytical upper bound on the optimal importance weight for the given privacy profile.

    Args:
        epsilon_1: The privacy loss when the importance weight is equal to 1.
        diff_1: The derivative of the privacy loss when the importance weight is equal to 1.
        target_epsilon: The epsilon parameter of the target privacy guarantee. Must be greater than or equal to epsilon_1.
        strong_convexity: The strong convexity of exp(privacy_loss(w)).

    Returns:
        The upper bound on the optimal importance weight.
    """
    v1 = -np.exp(epsilon_1)
    v2 = strong_convexity / 2 - diff_1 * np.exp(epsilon_1) - 1
    v_max = max(v1, v2)
    bracket = 2 * (target_epsilon + v_max) / strong_convexity + 1
    return bracket


def optimal_importance_weight(privacy_profile, diff_profile, target_epsilon, strong_convexity=None, tol=1e-6):
    """Compute optimal importance weight for a data-dependent privacy profile and target privacy level.

    Args:
        privacy_profile: A function that takes an importance weight and returns the privacy loss. Must be increasing and differentiable. exp(privacy_profile(w)) must be strongly convex.
        diff_profile: The derivative of privacy_profile.
        target_epsilon: The epsilon parameter of the target privacy guarantee. Must be greater than or equal to privacy_profile(1).
        strong_convexity: The strong convexity parameter of exp(private_loss(w)). If not provided, the function will be slower.
        tol: The tolerance in terms of absolute error.
    """
    epsilon_1 = privacy_profile(1)
    if target_epsilon < epsilon_1:
        logging.warn(f"target_epsilon must be greater than or equal to epsilon_1. target_epsilon={target_epsilon} and epsilon_1={epsilon_1}")
    diff_1 = diff_profile(1)
    if strong_convexity is None:
        upper_bracket = _upper_bracket_brute_force(privacy_profile, target_epsilon)
    else:
        upper_bracket = _upper_bracket_analytical(epsilon_1, diff_1, target_epsilon, strong_convexity)
    lower_bracket = 1
    if lower_bracket == upper_bracket:
        return lower_bracket
    n_iterations = np.ceil(np.log2(upper_bracket - lower_bracket) - np.log2(tol) - np.log2(2))

    def log_sampled_privacy(w):
        return logexpm1(privacy_profile(w)) - np.log(w)

    log_target = logexpm1(target_epsilon)
    for _ in range(int(n_iterations)):
        w = (lower_bracket + upper_bracket) / 2
        log_epsilon_w = log_sampled_privacy(w)
        if log_epsilon_w >= log_target:
            upper_bracket = w
        else:
            lower_bracket = w
    return (lower_bracket + upper_bracket) / 2


def optimal_importance_weight_fast(privacy_profile, diff_profile, target_epsilon, strong_convexity=None, xtol=None, rtol=1e-3):
    """Compute optimal importance weight for a data-dependent privacy profile and target privacy level.

    Args:
        privacy_profile: A function that takes an importance weight and returns the privacy loss. Must be increasing and differentiable. exp(privacy_profile(w)) must be strongly convex.
        diff_profile: The derivative of privacy_profile.
        target_epsilon: The epsilon parameter of the target privacy guarantee. Must be greater than or equal to privacy_profile(1).
        strong_convexity: The strong convexity parameter of exp(private_loss(w)). If not provided, the function will be slower.
        xtol: The tolerance in terms of absolute error.
        rtol: The tolerance in terms of relative error.
    """
    log_target = logexpm1(target_epsilon)

    def objective(w):
        return logexpm1(privacy_profile(w)) - np.log(w) - log_target

    if objective(1) > 0:
        raise ValueError("The chosen target_epsilon is not satisfiable, even without sampling.")

    lb, ub = 1., 2.
    while objective(ub) < 0:
        lb, ub = ub, 2. * ub

    result = sp_optimize.root_scalar(objective, bracket=[lb, ub], xtol=xtol, rtol=rtol)
    if not result.converged:
        raise ValueError(result)
    return result.root


class LloydProfile:
    """Privacy profile of the weighted Lloyd algorithm with Laplace noise.

    Args:
        lp_norm: The lp-norm of x.
        beta_count: The scale parameter of the Laplace noise for the count query.
        beta_sum: The scale parameter of the Laplace noise for the sum query.
        num_iterations: The number of iterations of the Lloyd algorithm.
    """
    def __init__(self, lp_norm: float, beta_count: float, beta_sum: float, num_iterations: int):
        self.lp_norm = lp_norm
        self.beta_count = beta_count
        self.beta_sum = beta_sum
        self.num_iterations = num_iterations

    @property
    def strong_convexity(self):
        factor_1 = 1 / self.beta_count + self.lp_norm / self.beta_sum
        exponent = self.num_iterations * factor_1
        return self.num_iterations ** 2 * factor_1 * np.exp(exponent)

    def __call__(self, w):
        """Compute the privacy loss for a given importance weight.

        Args:
            w: The importance weight.

        Returns:
            The privacy loss.
        """
        return self.num_iterations * w * (1 / self.beta_count + self.lp_norm / self.beta_sum)

    def diff(self, w):
        """Compute the derivative of the privacy loss for a given importance weight.

        Args:
            w: The importance weight.

        Returns:
            The derivative of the privacy loss.
        """
        return self.num_iterations * (1 / self.beta_count + self.lp_norm / self.beta_sum)


def privacy_optimal_weights_for_epsilon(X, T, beta_sum, beta_count, target_epsilon, norm, **kwargs):
    """Compute privacy-optimal weights for a given target_epsilon.

    Args:
        X: The data matrix.
        T: Number of DP-k-means iterations.
        beta_sum: The scale parameter for the sum noise.
        beta_count: The scale parameter for the count noise.
        target_epsilon: The target epsilon that should be achieved.
        norm: The norm for the sampling distribution.
    """
    weights = []
    for x in X:
        profile = LloydProfile(np.linalg.norm(x, ord=norm), beta_count, beta_sum, T)
        weights.append(optimal_importance_weight_fast(profile, profile.diff,
                                                      target_epsilon, strong_convexity=None,
                                                      **kwargs))
    weights = np.array(weights)
    q = 1/weights
    m = q.sum()
    return q, weights, m


def privacy_optimal_weights_for_m_fast(X, T, beta_sum, beta_count, target_m, norm, xtol=None, rtol=None, max_iter=None):
    """Find the target epsilon such that the expected sample size is close to target_M.

    Args:
        X: The data matrix.
        T: Number of DP-k-means iterations.
        beta_sum: The scale parameter for the sum noise.
        beta_count: The scale parameter for the count noise.
        target_m: The target expected sample size.
        norm: The norm for the sampling distribution.
        xtol: The tolerance in epsilon in terms of absolute error.
        rtol: The tolerance in epsilon terms of relative error.
        max_iter: The maximum number of epsilon candidates to check.

    Returns:
        q_opt: The optimal sampling probabilities.
        w_opt: The optimal importance weights.
        epsilon_opt: The optimal epsilon."""
    X_lp = np.linalg.norm(X, ord=norm, axis=1)
    n_points, _ = X.shape
    pdp_epsilons = LloydProfile(X_lp, beta_count, beta_sum, T)(1.0)
    pdp_max = pdp_epsilons.max()
    def m_residual(target_epsilon):
        if target_epsilon < pdp_max:
            return target_m + (pdp_max - target_epsilon)
        ws = pais_rs.all_optimal_weights(X_lp,
                                         beta_count=beta_count,
                                         beta_sum=beta_sum,
                                         num_iterations=T,
                                         target_epsilon=target_epsilon,)
        qs = 1 / np.array(ws)
        m = qs.sum()
        return m - target_m

    if m_residual(pdp_max) < 0:
        raise ValueError("The specified target_m is never optimal for the given hyperparameters. Consider decreasing it.")

    ub = pdp_max * 2
    while m_residual(ub) > 0:
        ub *= 2

    result = sp_optimize.root_scalar(m_residual, bracket=[pdp_max, ub], xtol=xtol, rtol=rtol, maxiter=max_iter)
    if not result.converged:
        raise ValueError(result)
    epsilon_opt = result.root
    q_opt, w_opt, _ = privacy_optimal_weights_for_epsilon(X, T, beta_sum, beta_count, epsilon_opt, norm)
    return q_opt, w_opt, epsilon_opt


def get_privacy_optimal_weights(X, T, beta_sum, beta_count, target_M, norm, it=10):
    logging.warn("Deprecated! Use privacy_optimal_weights_for_m_fast() instead.")

    r = np.linalg.norm(X, ord=norm, axis=1).max()
    A1 = T*(1/beta_count + r/beta_sum)

    # set initial bracket limits
    bracket_low = A1
    bracket_up = A1*2

    # compute a better upper limit
    # TODO: should that be also done for the lower one?
    for _ in range(100):
        target_epsilon = bracket_up
        weights = []
        for x in X:
            profile = LloydProfile(np.linalg.norm(x, ord=norm), beta_count, beta_sum, T)
            weights.append(optimal_importance_weight(profile, profile.diff,
                                                     target_epsilon, strong_convexity=None,
                                                     tol=1e-6))
        q = 1/np.array(weights)
        M = q.sum()
        if M<target_M:
            break
        else:
            bracket_low = bracket_up
            bracket_up = bracket_up * 2

    # conduct a bracket search for it iterations
    # TODO: break out if reasonably close
    for _ in range(it):
        # compute the M for a candidate epsilon which is the mid of the bracket
        target_epsilon = (bracket_low+bracket_up)/2
        weights = []
        for x in X:
            profile = LloydProfile(np.linalg.norm(x, ord=norm), beta_count, beta_sum, T)
            weights.append(optimal_importance_weight(profile, profile.diff,
                                                     target_epsilon, strong_convexity=None,
                                                     tol=1e-6))
        weights = np.array(weights)
        q = 1/weights
        M = q.sum()
        # update bracket limit
        if M > target_M:
            bracket_low = target_epsilon
        else:
            bracket_up = target_epsilon

    return q, weights, target_epsilon


def find_B_for_target_m(eps, d, m, norm, r, T, X_lp, lower_B=1e-7, upper_B=5e4, shrinkage_factor=0.5, max_it=250):
    def m_residual(B):
        # compute noise scales for B
        beta_sum, beta_count = utils.compute_noise_scales(B, d, r, T)
        # compute privacy-optimal weights
        weights_opt = pais_rs.all_optimal_weights(X_lp, beta_count, beta_sum, T, eps)
        weights_opt = np.array(weights_opt)
        # compute m based on weights
        m_opt = np.sum(1/weights_opt)
        return m_opt-m

    # find a working upper bound of B
    for _ in range(max_it):
        try:
            B = m_residual(upper_B)
        except:
            print(f'upper_B={upper_B} is too large')
            upper_B *= shrinkage_factor
            print(f'set upper_B={upper_B}')
        else:
            break

    # find B for target m
    B = sp_optimize.bisect(m_residual, lower_B, upper_B)

    return B
