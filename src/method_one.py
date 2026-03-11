
from scipy import stats
import numpy as np
import pandas as pd

from utils import forward_pass, generate_new_lambda, compute_log_joint, get_log_acceptance

def generate_new_N(sites: int, S: int, rng: np.random):
    return rng.integers(low=1, high=S, size=sites)

def run_method_one(sites, T, lam, p, S, EPOCHS, random_state=42) -> None:
    """Uniformly sample lambda, p, and N_i."""
    print("\n\nRunning method one")
    rng = np.random.default_rng(random_state)
    N, C = forward_pass(sites=sites, T=T, p=p, lam=lam, rng=rng)

    

    # Storing true values
    true_N = N
    true_p = p
    true_lam = lam

    # print("TRUE VALUES")
    # print("N:\n", N)
    # print("C:\n", C)
    # print("Lambda: ", lam)
    # print("p: \t", p)

    N_samples = []
    lam_samples = []
    p_samples = []

    burn_in = 1000

    # print("EPOCHS:", EPOCHS)
    num_accepted = 0
    
    # TODO: Add a time tracker. Also, it may look good to create some type of visuals for how lambda, p, and N change per iteration. 
    for i in range(EPOCHS):
        if i == 0:
            lam = lam # generate_new_lambda(S, rng)
            p = p # rng.uniform(0, 1)
            N = generate_new_N(sites, S, rng)

        log_old_joint = compute_log_joint(N, C, lam, p, S)

        new_lam = lam # generate_new_lambda(S, rng)
        new_p = p # rng.uniform(0, 1)
        new_N = generate_new_N(sites, S, rng)

        log_new_joint = compute_log_joint(new_N, C, new_lam, new_p, S)

        log_trans_new_to_old = np.sum(stats.poisson.logpmf(N, mu=new_lam))
        log_trans_old_to_new = np.sum(stats.poisson.logpmf(new_N, mu=lam))

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

    if not num_accepted: 
        print("No samples accepted.")
        return

    N_mean_comparison = pd.DataFrame({
        "Sites": [_ for _ in range(1, sites+1)],
        "True_N": true_N,
        "Mean_N": np.mean(N_samples, axis=0)
    })

    print(N_mean_comparison.to_string(index=False))
    print("Average Absolute Error in N estimation: ", np.abs(np.mean((true_N - np.mean(N_samples, axis=0)))))

    print(f"True Lambda: {true_lam} \t est. lam: {np.mean(lam_samples)}")
    print(f"True p: {true_p} \t\t est. p: {np.mean(p_samples)}")
    print(f"Samples accepted: {num_accepted}")


def main():
    run_method_one()

if __name__ == "__main__":
    main()