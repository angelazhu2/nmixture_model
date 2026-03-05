from scipy import stats
import numpy as np
import random
import pandas as pd

RANDOM_STATE = 0
rng = np.random.default_rng(RANDOM_STATE)

def forward_pass(sites: int, T: int, p: float, lam: int) -> tuple[np.ndarray, np.ndarray]:
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
    return log_lam + log_p + np.sum(np.sum(log_C, axis=1) + log_N)

def generate_new_lambda(S: int):
    return rng.uniform(1, S)

def get_acceptance_prob(joint_ratio: float, proposal_ratio: float):
    return np.minimum(1, joint_ratio*proposal_ratio)

def get_log_acceptance(log_old_joint, log_new_joint, log_trans_new_to_old, log_trans_old_to_new):
    """https://acme.byu.edu/00000186-a3cf-d653-a78f-fbdfc8620001/metropolis-pdf#:~:text=Your%20function%20should%20return%20an,%3E%3E%3E%20m%20=%2080"""
    ratio = (log_new_joint - log_old_joint) + (log_trans_new_to_old - log_trans_old_to_new)
    return np.minimum(0, ratio)

def main() -> None:
    sites = 10
    T = 5
    lam = 20 
    p = 0.4
    S = 40

    N, C = forward_pass(sites=sites, T=T, p=p, lam=lam)
    # N = N.item()
    # C = C.item()

    # Storing true values
    true_N = N
    true_p = p
    true_lam = lam

    # For a single site and time
    print("TRUE VALUES")
    print("N:\n", N)
    print("C:\n", C)
    print("Lambda: ", lam)
    print("p: \t", p)

    N_samples = []
    lam_samples = []
    p_samples = []

    burn_in = 0
    EPOCHS = 10000
    print("EPOCHS:", EPOCHS)
    num_accepted = 0
    # BACKWARD PASS
    for i in range(EPOCHS):
        if i == 0:
            lam = generate_new_lambda(S)
            p = rng.uniform(0, 1)
            N = rng.poisson(lam, size=sites)

        log_old_joint = compute_log_joint(N, C, lam, p, S)

        new_lam = generate_new_lambda(S)
        new_p = rng.uniform(0, 1)
        new_N = rng.poisson(new_lam, size=sites)

        log_new_joint = compute_log_joint(new_N, C, new_lam, new_p, S)

        log_trans_new_to_old = np.sum(stats.poisson.logpmf(N, mu=new_lam))
        log_trans_old_to_new = np.sum(stats.poisson.logpmf(new_N, mu=lam))

        acceptance_score = get_log_acceptance(log_old_joint, log_new_joint, log_trans_new_to_old, log_trans_old_to_new)

        U = np.log(rng.uniform())
        if acceptance_score >= U:
            num_accepted += 1
            # Update params
            N: np.ndarray = new_N
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

if __name__ == "__main__":
    main()

