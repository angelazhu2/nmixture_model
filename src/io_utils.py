from __future__ import annotations
import numpy as np
import pandas as pd


def save_simulated_data(root: str, N: np.ndarray, C: np.ndarray) -> None:
    path = f"{root}/simulated_data_N_{len(N)}_C_{C.shape[1]}.csv"
    df = pd.DataFrame(C, columns=[f"C_{t+1}" for t in range(C.shape[1])])
    df.insert(0, "N", N)
    df.to_csv(path, index=False)
    print(f"Simulated data saved to {path}")


def save_samples(root, method, true_lam, true_p, true_avg_N, N_samples, lam_samples, p_samples):
    path = f"{root}/samples_{true_lam}lamda_{true_p}p_{true_avg_N}avgN_method{method}.csv"
    df = pd.DataFrame({
        "true_lambda": true_lam,
        "lambda": lam_samples,
        "true_p": true_p,
        "p": p_samples,
        "true_avg_N": true_avg_N,
        "avg_N": [np.mean(n) for n in N_samples],
        "N": N_samples,
    })
    df.to_csv(path, index=False)
    print(f"Samples saved to {path}")
