from scipy import stats
import numpy as np
import random
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE) 

def forward_pass(sites: int, T: int, p: float, lam: int) -> tuple[np.ndarray, np.ndarray]:
    N = stats.poisson(mu=lam).rvs(size=sites, random_state=RANDOM_STATE) #true abundance per site
    C = np.zeros((sites, T)) #observed counts per site and survey
    for t in range(T):
        C[:, t] = stats.binom(n=N, p=p).rvs(size=sites, random_state=RANDOM_STATE)
    return N, C

def compute_joint(N: int, C: int, lam: int, p: float, lam_uniform_size: int) -> float:
    p_C = stats.binom.pmf(C, N, p)
    p_N = stats.poisson.pmf(N, lam)
    p_lam = 1/lam_uniform_size
    p_p = stats.uniform.pdf(p, loc=0, scale=1)
    p_joint = p_C * p_N * p_lam * p_p
    return p_joint

def generate_new_lambda(S: int):
    x = [_ for _ in range(S+1)]
    rand_idx = random.randint(1, S)
    return x[rand_idx]

def get_acceptance_prob(joint_ratio: float, proposal_ratio: float):
    return np.minimum(1, joint_ratio*proposal_ratio)

def main() -> None:
    sites = 1
    T = 1
    lam = 20 
    p = 0.4 
    S = 40

    N, C = forward_pass(sites=sites, T=T, p=p, lam=lam)
    N = N.item()
    C = C.item()

    # Storing true values
    true_N = N
    true_p = p
    true_lam = lam

    # For a single site and time
    print("VALUES")
    print("N: \t", N)
    print("C: \t", C)
    print("Lambda: ", lam)
    print("p: \t", p)

    sample_cols = ["N", "C", "Lambda", "p"]
    samples = []

    burn_in = 0
    EPOCHS = 1000000
    # BACKWARD PASS
    for i in range(EPOCHS):
        p_old_joint = compute_joint(N, C, lam, p, S)

        new_lam = generate_new_lambda(S)
        new_p = np.random.uniform(low=0, high=1, size=1).item()
        new_N = np.random.poisson(lam=new_lam, size=1).item()

        p_new_joint = compute_joint(new_N, C, new_lam, new_p, S)   

        prop_dist_new_to_old = stats.poisson.pmf(N, mu=new_lam)
        prop_dist_old_to_new = stats.poisson.pmf(new_N, mu=lam)

        joint_ratio = p_new_joint / p_old_joint
        proposal_ratio = prop_dist_new_to_old / prop_dist_old_to_new

        acceptance = get_acceptance_prob(joint_ratio, proposal_ratio)

        # Collect sample
        U = np.random.uniform()
        if acceptance >= U:
            # Update params
            N = new_N
            p = new_p
            lam = new_lam

        if burn_in < i:
            samples.append([N, C, lam, p])

    samples = pd.DataFrame(samples, columns=sample_cols)

    sample_N_mean = samples["N"].mean()
    sample_lambda_mean = samples["Lambda"].mean()
    sample_p_mean = samples["p"].mean()

    print(f"True N: {true_N} \t\t Estimates N: {sample_N_mean:.5f}")
    print(f"True lambda: {true_lam} \t Estimates lambda: {sample_lambda_mean:.5f}")
    print(f"True p: {true_p} \t\t Estimates p: {sample_p_mean:.5f}")

if __name__ == "__main__":
    main()

