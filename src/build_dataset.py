import numpy as np
from scipy import stats

from utils import save_to_csv, forward_pass



#from textbook
def forward_pass_txt():
    sites = 150
    J = 2 #abundance measurement per site

    lambda_i = 2.5 #expected abundance per site
    p = 0.4 #detection probability (per individual)

    N = stats.poisson(mu=lambda_i).rvs(size=sites) #true abundance per site

    C = np.zeros((sites, J)) #observed counts per site and survey

    for j in range(J):
        C[:, j] = stats.binom(n=N, p=p).rvs(size=sites) #observed count per site and survey

    print(N.shape)
    print(C.shape)

    # True abundance Data
    values, bincounts = np.unique(N, return_counts=True)
    print("table of N")
    print(values)
    print(bincounts)
    print(f"True total population size:\n{np.sum(N)}")
    print(f"True mean abundance per site:\n{np.mean(N)}")

    # Observed counts Data
    # First 6 sites with counts for each survey
    print("\nFirst 6 sites - observed counts for each survey:")
    print("Site | True Abundance | Survey 1 | Survey 2")
    print("-----|----------------|----------|----------")
    for i in range(6):
        print(f" {i+1:2d}  |       {int(N[i]):2d}       |    {int(C[i, 0]):2d}    |    {int(C[i, 1]):2d}")

    observed_abundance = np.max(C, axis=1)
    print("\ntable of C")
    values, bincounts = np.unique(observed_abundance, return_counts=True)
    print(values)
    print(bincounts)
    print("total observed population size:\n", np.sum(observed_abundance))
    print("mean observed abundance per site:\n", np.mean(observed_abundance))
    return N, C


def main():
    random_state = 42
    rng = np.random.default_rng(random_state)
    sites = 20
    T = 6
    lam = 2
    p = 0.25
    
    N, C = forward_pass(sites=sites, T=T, p=p, lam=lam, rng=rng)
    save_to_csv("../data/simulated_data.csv", N, C)


if __name__ == "__main__":
    main()