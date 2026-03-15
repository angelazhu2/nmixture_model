# accept parameters one at a time
# metropolis within gibs

from __future__ import annotations

import numpy as np
from scipy import stats
import pandas as pd
import time

from utils import forward_pass, generate_new_lambda, get_log_acceptance, compute_log_joint
from io_utils import save_samples, save_summary

def generate_new_N(sites: int, S: int, rng: np.random):
    return rng.integers(low=1, high=S, size=sites)

def run_method_four(sites, T, lam, p, S, EPOCHS, random_state=42) -> None:
    """Random walk method (component-wise)"""
    print("\n\nRunning method four (component-wise).")
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
    N = rng.poisson(lam, size=sites) # generate_new_N(sites, S, rng)
    N = np.maximum(N, C_max)

    for i in range(EPOCHS):
        #import pdb; pdb.set_trace()
        # propose lambda 
        temp_lam = None
        counter = 0
        while True: 
            if counter == 5:
                break 
            counter += 1   
            while True: 
                temp_lam = rng.normal(loc=lam, scale=1, size=1).item()
                if temp_lam > 0:
                    new_lam = temp_lam
                    break
            
            log_new = compute_log_joint(N, C, new_lam, p, S)
            log_old = compute_log_joint(N, C, lam, p, S)

            acceptance_score = get_log_acceptance(log_old, log_new, 0, 0)
            if np.log(rng.uniform()) <= acceptance_score:
                lam = new_lam
                break
        
        # propose p 
        temp_p = None
        counter = 0
        while True: 
            if counter == 5:
                break 
            counter += 1   
            while True:
                temp_p = rng.normal(loc=p, scale=0.1)
                if 0 < temp_p < 1:
                    new_p = temp_p
                    break
            log_new = compute_log_joint(N, C, lam, new_p, S)
            log_old = compute_log_joint(N, C, lam, p, S)

            acceptance_score = get_log_acceptance(log_old, log_new, 0, 0)
            if np.log(rng.uniform()) <= acceptance_score:
                p = new_p
                break

        new_N = N.copy()
        # propose N 
        for j in range(sites):
            counter = 0
            while True:
                if counter == 5:
                    break 
                counter += 1   

                proposed_N_i = rng.poisson(N[j])
                new_N[j] = np.maximum(proposed_N_i, C_max[j])

                log_new = compute_log_joint(new_N, C, lam, p, S)
                log_old = compute_log_joint(N, C, lam, p, S)

                log_trans_new_to_old = np.sum(stats.poisson.logpmf(N, mu=new_N))
                log_trans_old_to_new = np.sum(stats.poisson.logpmf(new_N, mu=N))

                acceptance_score = get_log_acceptance(log_old, log_new, log_trans_new_to_old, log_trans_old_to_new)
                if np.log(rng.uniform()) <= acceptance_score:
                    N[j] = new_N[j]
                    break 

        if burn_in < i:
            num_accepted += 1
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

    save_samples("../data/results", method=4,
                 sites=sites, T=T, S=S, EPOCHS=EPOCHS,
                 true_lam=true_lam, true_p=true_p,
                 N_samples=N_samples, lam_samples=lam_samples, p_samples=p_samples, total_time=total_time, prop_sparsity=prop_sparsity)
    save_summary("../data/results", method=4,
                 sites=sites, T=T, S=S, EPOCHS=EPOCHS,
                 true_lam=true_lam, true_p=true_p, true_N=true_N,
                 lam_samples=lam_samples, p_samples=p_samples, N_samples=N_samples,
                 num_accepted=num_accepted, acceptance_rate=acceptance_rate,
                 total_time=total_time, prop_sparsity=prop_sparsity)