from __future__ import annotations

import numpy as np
from scipy import stats
import pandas as pd

from utils import forward_pass, generate_new_lambda, get_log_acceptance, compute_log_joint
from io_utils import save_samples, save_summary

def generate_new_N(sites: int, S: int, rng: np.random):
    return rng.integers(low=1, high=S, size=sites)

def run_method_three(sites, T, lam, p, S, EPOCHS, random_state=42) -> None:
    """Random walk method."""
    print("\n\nRunning method three")
    rng = np.random.default_rng(random_state)
    N, C = forward_pass(sites=sites, T=T, p=p, lam=lam, rng=rng)

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
    lam = generate_new_lambda(S, rng)
    p = rng.uniform()
    N = rng.poisson(lam, size=sites)# generate_new_N(sites, S, rng)
    # N = np.maximum(N, C_max)
    log_old_joint = compute_log_joint(N, C, lam, p, S)

    for i in range(EPOCHS):
        new_lam = None
        while True: 
            temp = rng.normal(loc=lam, scale=1, size=1).item()
            if temp > 0:
                new_lam = temp
                break
        new_p = None
        while True:
            temp = rng.normal(loc=p, scale=0.1)
            if 0 < temp < 1:
                new_p = temp
                break
        
        new_N = np.zeros(shape=N.shape)
        for j in range(sites):
            while True:
                proposed_N_i = rng.poisson(N[j])
                if proposed_N_i >= C_max[j]:
                    break
            new_N[j] = proposed_N_i
    
        log_new_joint = compute_log_joint(new_N, C, new_lam, new_p, S)

        log_trans_new_to_old = np.sum(stats.poisson.logpmf(N, mu=new_N))
        log_trans_old_to_new = np.sum(stats.poisson.logpmf(new_N, mu=N))

        acceptance_score = get_log_acceptance(log_old_joint, log_new_joint, log_trans_new_to_old, log_trans_old_to_new)

        U = np.log(rng.uniform())
        if acceptance_score >= U: 
            log_old_joint = log_new_joint
            N = new_N # true_N 
            p = new_p
            lam = new_lam # true_lam 

            if burn_in < i:
                num_accepted += 1
                N_samples.append(N.tolist())
                lam_samples.append(lam)
                p_samples.append(p)

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

    save_samples("../data/results", method=3,
                 true_lam=true_lam, true_p=true_p,
                 N_samples=N_samples, lam_samples=lam_samples, p_samples=p_samples)
    save_summary("../data/results", method=3,
                 true_lam=true_lam, true_p=true_p, true_N=true_N,
                 lam_samples=lam_samples, p_samples=p_samples, N_samples=N_samples,
                 num_accepted=num_accepted, acceptance_rate=acceptance_rate)