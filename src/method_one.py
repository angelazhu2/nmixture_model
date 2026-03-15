from __future__ import annotations
import numpy as np
import pandas as pd
import time

from utils import forward_pass, compute_log_joint, get_log_acceptance
from io_utils import save_samples, save_summary

def generate_new_N(sites: int, C_max:np.ndarray, S: int, rng: np.random):

    return rng.integers(low=C_max, high=S, size=sites)

def run_method_one(sites, T, lam, p, S, EPOCHS, random_state=42) -> None:
    """Uniformly sample lambda, p, and N_i."""
    print("\n\nRunning method one")
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
    epsilon = 1e-4
    
    start = time.perf_counter()

    C_max = np.max(C, axis=1)
    S = int(np.maximum(S, C_max.max()))
    lam = rng.uniform(1, S)
    p = rng.uniform(epsilon, 1-epsilon)
    N = rng.integers(low=C_max, high=S, size=sites)

    # TODO: Add a time tracker. Also, it may look good to create some type of visuals for how lambda, p, and N change per iteration. 

    for i in range(EPOCHS):
        log_old_joint = compute_log_joint(N, C, lam, p, S)
        # print(log_old_joint)
        new_lam = rng.uniform(1, S)
        new_p = rng.uniform(epsilon, 1-epsilon)
        new_N = rng.integers(low=C_max, high=S, size=sites) #generate_new_N(sites, S, rng)

        log_new_joint = compute_log_joint(new_N, C, new_lam, new_p, S)

        acceptance_score = get_log_acceptance(log_old_joint, log_new_joint, 0, 0)

        U = np.log(rng.uniform())
        if acceptance_score >= U: 
            num_accepted += 1
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

    save_samples("../data/results", method=1,
                 sites=sites, T=T, S=S, EPOCHS=EPOCHS,
                 true_lam=true_lam, true_p=true_p,
                 N_samples=N_samples, lam_samples=lam_samples, p_samples=p_samples, total_time=total_time, prop_sparsity=prop_sparsity)
    save_summary("../data/results", method=1,
                 sites=sites, T=T, S=S, EPOCHS=EPOCHS,
                 true_lam=true_lam, true_p=true_p, true_N=true_N,
                 lam_samples=lam_samples, p_samples=p_samples, N_samples=N_samples,
                 num_accepted=num_accepted, acceptance_rate=acceptance_rate,
                 total_time=total_time, prop_sparsity=prop_sparsity)

def main():
    run_method_one()

if __name__ == "__main__":
    main()