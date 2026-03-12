from __future__ import annotations
import numpy as np
from scipy import stats


def forward_pass(sites: int, T: int, p: float, lam: int, rng:np.random) -> tuple[np.ndarray, np.ndarray]:
    N: np.ndarray = rng.poisson(lam, size=sites) #true abundance per site
    C = np.zeros((sites, T)) #observed counts per site and survey
    for t in range(T):
        C[:, t] = rng.binomial(N, p, size=sites)
    return N.squeeze(), C.squeeze()

def compute_log_joint(N: np.ndarray, C: np.ndarray, lam: int, p: float, S: int) -> float:
    log_lam = stats.uniform.logpdf(lam, loc=1, scale=S-1)
    log_p = stats.uniform.logpdf(p, loc=0, scale=1)
    log_C = np.zeros(C.shape)
    for t in range(C.shape[1]):
        log_C[:, t] = stats.binom.logpmf(C[:, t], N, p)
    log_N = stats.poisson.logpmf(N, lam)
    return np.sum(log_N + np.sum(log_C, axis=1), axis=0)

def generate_new_lambda(S: int, rng:np.random):
    return rng.uniform(1, S)

def get_acceptance_prob(joint_ratio: float, proposal_ratio: float):
    return np.minimum(1, joint_ratio*proposal_ratio)

def get_log_acceptance(log_old_joint, log_new_joint, log_trans_new_to_old, log_trans_old_to_new):
    """https://acme.byu.edu/00000186-a3cf-d653-a78f-fbdfc8620001/metropolis-pdf#:~:text=Your%20function%20should%20return%20an,%3E%3E%3E%20m%20=%2080"""
    if not np.isfinite(log_new_joint):
        return -np.inf
    ratio = (log_new_joint - log_old_joint) + (log_trans_new_to_old - log_trans_old_to_new)
    return np.minimum(0, ratio)