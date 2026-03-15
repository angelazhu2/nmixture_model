from __future__ import annotations
import numpy as np
import pandas as pd

def save_simulated_data(root: str, N: np.ndarray, C: np.ndarray) -> None:
    path = f"{root}/simulated_data_N_{len(N)}_C_{C.shape[1]}.csv"
    df = pd.DataFrame(C, columns=[f"C_{t+1}" for t in range(C.shape[1])])
    df.insert(0, "N", N)
    df.to_csv(path, index=False)
    print(f"Simulated data saved to {path}")


def save_samples(root, method, sites, T, S, EPOCHS, true_lam, true_p, N_samples, lam_samples, p_samples, total_time, prop_sparsity):
    path = f"{root}/mthd{method}_sites{sites}_T{T}_S{S}_EPOCHS{EPOCHS}_lam{true_lam}_p{true_p}_runs.csv"
    N_arr = np.array(N_samples)
    df = pd.DataFrame({
        "lambda": lam_samples,
        "p": p_samples,
        "avg_N": N_arr.mean(axis=1),
        "total_N": N_arr.sum(axis=1),
        "total_time": total_time,
        "prop_sparsity": prop_sparsity
    })
    for s in range(N_arr.shape[1]):
        df[f"N_{s+1}"] = N_arr[:, s]
    df.to_csv(path, index=False)
    print(f"Samples saved to {path}")


def save_summary(root, method, sites, T, S, EPOCHS, true_lam, true_p, true_N, lam_samples, p_samples, N_samples, num_accepted, acceptance_rate, total_time, prop_sparsity):
    N_arr = np.array(N_samples)
    mean_N = N_arr.mean(axis=0)
    path = f"{root}/method{method}_summary.csv"
    with open(path, 'a') as f:
        pd.Series({
            "Sites": sites, 
            "visits": T, 
            "bound on lambda": S,
            "Epochs": EPOCHS,
            "true_lam": true_lam,
            "est_lam": np.mean(lam_samples),
            "true_p": true_p,
            "est_p": np.mean(p_samples),
            "true_avg_N": np.mean(true_N),
            "est_avg_N": mean_N.mean(),
            "true_total_N": int(np.sum(true_N)),
            "est_total_N": np.sum(mean_N),
            "N_mae": float(np.mean(np.abs(true_N - mean_N))),
            "num_accepted": num_accepted,
            "acceptance_rate": acceptance_rate,
            "total_time": total_time,
            "prop_sparsity": prop_sparsity
        }).to_csv(f, header=["--- NEW RUN ---"])
    print(f"Summary saved to {path}")
