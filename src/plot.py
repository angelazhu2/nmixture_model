from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from data import get_data

def filter_baseline(df: pd.DataFrame, is_method4: bool) -> pd.DataFrame:
    # Baseline params
    sites = 20
    T = 6
    lam = 5
    p = 0.25
    if not is_method4:
        EPOCHS = 40000
        return df[
            (df["Sites"] == sites) &
            (df["visits"] == T) &
            (df["true_lam"] == lam) &
            (df["true_p"] == p) &
            (df["Epochs"] == EPOCHS)]
    else:
        EPOCHS = 10000
        return df[
            (df["Sites"] == sites) &
            (df["visits"] == T) &
            (df["true_lam"] == lam) &
            (df["true_p"] == p) &
            (df["Epochs"] == EPOCHS)]


def main():
    method1_df: pd.DataFrame = get_data("method1")
    method2_df: pd.DataFrame = get_data("method2")
    method3_df: pd.DataFrame = get_data("method3")
    method4_df: pd.DataFrame = get_data("method4")
    method5_df: pd.DataFrame = get_data("method5")

    print(method1_df.shape)
    print(method2_df.shape)
    print(method3_df.shape)
    print(method4_df.shape)
    print(method5_df.shape)

    print(method1_df.columns)

    method_names = ["Method 1", "Method 2", "Method 3", "Method 4", "Method 5"]

    dfs = [method1_df, method2_df, method3_df, method4_df, method5_df]
    dfs = [filter_baseline(df, is_method4=(i == 3)) for i, df in enumerate(dfs)]
    for df in dfs:
        df["N_diff"] = abs(df["true_avg_N"] - df["est_avg_N"])

    avg_p_diffs = [np.mean(df["N_diff"]) for df in dfs]

    plt.figure(figsize=(8, 5))
    plt.bar(method_names, avg_p_diffs)
    plt.ylabel("Mean Absolute Difference in Estimated and True N")
    plt.title("Average Estimation Error in N by Method (baseline parameters)")
    plt.tight_layout()
    # plt.ylim(-6,6)
    plt.show()

if __name__ == "__main__":
    main()