from __future__ import annotations
import numpy as np
from scipy import stats
import pandas as pd

from utils import forward_pass, generate_new_lambda, compute_log_joint, get_log_acceptance
from io_utils import save_samples, save_summary

def run_method_two(sites, T, lam, p, S, EPOCHS, random_state=42) -> None:
    """Uniform lambda and p. Estimate N from poisson given lambda."""
    print("\n\nRunning method two")
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
    num_accepted = 0
    lam = generate_new_lambda(S, rng)
    p = rng.uniform()
    N = rng.poisson(lam, size=sites)

    # TODO: Add a time tracker. Also, it may look good to create some type of visuals for how lambda, p, and N change per iteration.
    for i in range(EPOCHS):
            
        log_old_joint = compute_log_joint(N, C, lam, p, S)

        new_lam = generate_new_lambda(S, rng)
        new_p = rng.uniform(0, 1)
        new_N = rng.poisson(new_lam, size=sites)

        log_new_joint = compute_log_joint(new_N, C, new_lam, new_p, S)

        # log_trans_new_to_old = np.sum(stats.poisson.logpmf(N, mu=new_lam))
        # log_trans_old_to_new = np.sum(stats.poisson.logpmf(new_N, mu=lam))

        log_trans_new_to_old = np.sum(stats.poisson.logpmf(N, mu=lam))
        log_trans_old_to_new = np.sum(stats.poisson.logpmf(new_N, mu=new_lam))
        
        acceptance_score = get_log_acceptance(log_old_joint, log_new_joint, log_trans_new_to_old, log_trans_old_to_new)

        U = rng.uniform()
        if np.exp(acceptance_score) >= U:
            num_accepted += 1
            N = new_N
            p = new_p
            lam = new_lam

        if burn_in < i:
            
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
        "Mean_N": mean_N,
    })

    print(N_mean_comparison.to_string(index=False))
    print(f"Avg true N: {np.mean(true_N)} \t\t est. avg N: {mean_N.mean()}")
    print("Average Absolute Error in N estimation: ", np.mean(np.abs(true_N - mean_N)))
    print(f"True Total Abundance: {np.sum(true_N)}")
    print(f"Estimated Total Abundance: {np.sum(mean_N)}")
    print(f"True Lambda: {true_lam} \t est. lam: {np.mean(lam_samples)}")
    print(f"True p: {true_p} \t\t est. p: {np.mean(p_samples)}")
    print(f"Samples accepted: {num_accepted}")
    print(f"Acceptance rate: {acceptance_rate:.3f}")

    save_samples("../data/results", method=2,
                 true_lam=true_lam, true_p=true_p,
                 N_samples=N_samples, lam_samples=lam_samples, p_samples=p_samples)
    save_summary("../data/results", method=2,
                 true_lam=true_lam, true_p=true_p, true_N=true_N,
                 lam_samples=lam_samples, p_samples=p_samples, N_samples=N_samples,
                 num_accepted=num_accepted, acceptance_rate=acceptance_rate)

def main():
    run_method_two()

if __name__ == "__main__":
    main()