from __future__ import annotations

import numpy as np
from scipy import stats
import pandas as pd
import time

from utils import forward_pass, generate_new_lambda, get_log_acceptance
from io_utils import save_samples, save_summary

def generate_new_N(sites: int, S: int, rng: np.random):
    return rng.integers(low=1, high=S, size=sites)

def compute_log_joint(N: np.ndarray, C: np.ndarray, lam: float, p: float, S: int) -> float:
    log_C = np.zeros(C.shape)
    for t in range(C.shape[1]):
        log_C[:, t] = stats.binom.logpmf(C[:, t], N, p)
    log_N = stats.poisson.logpmf(N, lam)
    joint = np.sum(log_N + np.sum(log_C, axis=1), axis=0)
    return joint

def log_truncated_poisson_pmf(x, mu, lower):
    log_numerator = stats.poisson.logpmf(x, mu)
    log_denominator = np.log(1 - stats.poisson.cdf(lower - 1, mu))
    return log_numerator - log_denominator

def standardize_bounds(loc: float, scale: float, lower: float, upper: float) -> tuple[float, float]:
    a, b = (lower - loc) / scale, (upper - loc) / scale
    return a, b

def run_method_five(sites, T, lam, p, S, EPOCHS, random_state=42) -> None:
    """Random walk method w/ truncated normal (method 3 reimplemented)."""
    print("\n\nRunning method five: Random walk method w/ truncated normal (method 3 reimplemented).")
    rng = np.random.default_rng(random_state)
    N, C = forward_pass(sites=sites, T=T, p=p, lam=lam, rng=rng)
    prop_sparsity = np.mean(C == 0)

    # Storing true values
    true_N = N
    true_p = p
    true_lam = lam

    N_samples = []
    lam_samples = []
    p_samples = []

    burn_in = EPOCHS // 4
    # print("EPOCHS:", EPOCHS)
    num_accepted = 0
    
    # TODO: Add a time tracker. Also, it may look good to create some type of visuals for how lambda, p, and N change per iteration. 
    # p = rng.uniform(0, 1)
    C_max = np.max(C, axis=1)
    epsilon = 1e-4
    start = time.perf_counter()

    S = int(np.maximum(S, C_max.max()))
    lam = generate_new_lambda(S, rng)
    p = rng.uniform(epsilon, 1-epsilon)
    N = rng.poisson(lam, size=sites)# generate_new_N(sites, S, rng)
    N = np.maximum(N, C_max)
    log_old_joint = compute_log_joint(N, C, lam, p, S)

    for i in range(EPOCHS):
        # Get new lambda
        lam_scale = 1
        lam_lower, lam_upper = standardize_bounds(loc=lam, scale=lam_scale, lower=1, upper=S)
        new_lam = stats.truncnorm.rvs(lam_lower, lam_upper, loc=lam, scale=lam_scale, size=1, random_state=rng).item()

        # Get new p
        epsilon = 1e-4
        p_scale = 0.1 
        p_lower, p_upper = standardize_bounds(loc=p, scale=p_scale, lower=epsilon, upper=1-epsilon)
        new_p = stats.truncnorm.rvs(p_lower, p_upper, loc=p, scale=p_scale, size=1, random_state=rng).item()

        # Get new N
        new_N = np.zeros(shape=N.shape)
        for j in range(sites):
            while True:
                proposed = rng.poisson(N[j])
                if proposed >= C_max[j]:
                    break
            new_N[j] = proposed
   
        log_new_joint = compute_log_joint(new_N, C, new_lam, new_p, S)

        # Handle N transition
        log_trans_old_to_new = np.sum([
            log_truncated_poisson_pmf(new_N[j], mu=N[j], lower=C_max[j]) 
            for j in range(sites)
        ])

        # Backward: probability of proposing old N given new_N, truncated to >= C_max
        log_trans_new_to_old = np.sum([
            log_truncated_poisson_pmf(N[j], mu=new_N[j], lower=C_max[j]) 
            for j in range(sites)
        ])

        # Handle lambda transition
        lower, upper = standardize_bounds(loc=new_lam, scale=lam_scale, lower=1, upper=S)
        log_trans_new_to_old += stats.truncnorm.logpdf(lam, lower, upper, loc=new_lam, scale=lam_scale)

        lower, upper = standardize_bounds(loc=lam, scale=lam_scale, lower=1, upper=S)
        log_trans_old_to_new += stats.truncnorm.logpdf(new_lam, lower, upper, loc=lam, scale=lam_scale)

        # Handle p transition
        lower, upper = standardize_bounds(loc=new_p, scale=p_scale, lower=epsilon, upper=1-epsilon)
        log_trans_new_to_old += stats.truncnorm.logpdf(p, lower, upper, loc=new_p, scale=p_scale)

        lower, upper = standardize_bounds(loc=p, scale=p_scale, lower=epsilon, upper=1-epsilon)
        log_trans_old_to_new += stats.truncnorm.logpdf(new_p, lower, upper, loc=p, scale=p_scale)
    

        acceptance_score = get_log_acceptance(log_old_joint, log_new_joint, log_trans_new_to_old, log_trans_old_to_new)

        U = np.log(rng.uniform())
        if acceptance_score >= U: 
            num_accepted += 1
            log_old_joint = log_new_joint
            N = new_N 
            p = new_p
            lam = new_lam  

        if burn_in < i:
            N_samples.append(N.tolist())
            lam_samples.append(lam)
            p_samples.append(p)

    end = time.perf_counter()
    total_time = np.round(end - start, 5)
    print(total_time, "seconds")

    if not num_accepted:
        print("No samples accepted.")
        return
    
    N_arr = np.array(N_samples)
    mean_N = N_arr.mean(axis=0)
    acceptance_rate = num_accepted / (EPOCHS - burn_in)

    N_mean_comparison = pd.DataFrame({
        "Sites": [_ for _ in range(1, sites+1)],
        "True_N": true_N,
        "Mean_N": mean_N
    })

    print(N_mean_comparison.to_string(index=False))
    print("Average Absolute Error in N estimation: ", np.mean(np.abs(true_N - mean_N)))

    print(f"True Total Abundance: {np.sum(true_N)}")
    print(f"Estimated Total Abundance: {np.sum(mean_N)}")

    print(f"True Lambda: {true_lam} \t est. lam: {np.mean(lam_samples)}")
    print(f"True p: {true_p} \t\t est. p: {np.mean(p_samples)}")
    print(f"Samples accepted: {num_accepted}")
    print(f"Proportion of Sparsity: {prop_sparsity}")

    save_samples("../data/results", method=5,
                 sites=sites, T=T, S=S, EPOCHS=EPOCHS,
                 true_lam=true_lam, true_p=true_p,
                 N_samples=N_samples, lam_samples=lam_samples, p_samples=p_samples, total_time=total_time, prop_sparsity=prop_sparsity)
    save_summary("../data/results", method=5,
                 sites=sites, T=T, S=S, EPOCHS=EPOCHS,
                 true_lam=true_lam, true_p=true_p, true_N=true_N,
                 lam_samples=lam_samples, p_samples=p_samples, N_samples=N_samples,
                 num_accepted=num_accepted, acceptance_rate=acceptance_rate,
                 total_time=total_time, prop_sparsity=prop_sparsity)