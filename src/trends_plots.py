import os
import pandas as pd
import matplotlib.pyplot as plt

METHODS = ["Method 1", "Method 2", "Method 3", "Method 4", "Method 5"]
COLORS = ["dodgerblue", "#F28500", "tab:green", "tomato", "tab:purple"]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
FILTER_VISITS = 6
TRUE_LAM = 5.0
TRUE_P = 0.25


def _load_data():
    frames = []
    for i, method in enumerate(METHODS, start=1):
        path = os.path.join(RESULTS_DIR, f"Method{i}.csv")
        df = pd.read_csv(path)
        df["method"] = method
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data = data[
        (data["visits"] == FILTER_VISITS)
        & (data["true_lam"] == TRUE_LAM)
        & (data["true_p"] == TRUE_P)
    ].sort_values("Sites").drop_duplicates(subset=["method", "Sites"])
    return data


DATA = _load_data()
SITES = sorted(DATA["Sites"].unique().tolist())
TRUE_TOTAL_N = DATA[DATA["method"] == METHODS[0]].sort_values("Sites")["true_total_N"].tolist()


def _series(col):
    return {
        method: DATA[DATA["method"] == method].sort_values("Sites")[col].tolist()
        for method in METHODS
    }


def _base_ax(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(SITES)
    ax.legend()


def plot_total_N():
    est_total_N = _series("est_total_N")
    _, ax = plt.subplots(figsize=(8, 5))
    ax.plot(SITES, TRUE_TOTAL_N, "k--", linewidth=2, label="True total N")
    for method, color in zip(METHODS, COLORS):
        ax.plot(SITES, est_total_N[method], marker="o", color=color, label=method)
    _base_ax(ax, "Number of Sites", "Total Population N",
             "Estimated Population over Number of Sites")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR,"total_N_by_sites.png"), dpi=150)
    plt.show()


def plot_n_mae():
    n_mae = _series("N_mae")
    _, ax = plt.subplots(figsize=(8, 5))
    for method, color in zip(METHODS, COLORS):
        ax.plot(SITES, n_mae[method], marker="o", color=color, label=method)
    _base_ax(ax, "Number of Sites", "Absolute Value of Mean Error at each site",
             "N Mean Error by Number of Sites")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR,"n_mae_by_sites.png"), dpi=150)
    plt.show()


def plot_lambda_error():
    est_lam = _series("est_lam")
    _, ax = plt.subplots(figsize=(8, 5))
    for method, color in zip(METHODS, COLORS):
        errors = [abs(e - TRUE_LAM) for e in est_lam[method]]
        ax.plot(SITES, errors, marker="o", color=color, label=method)
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    _base_ax(ax, "Number of Sites", "Absolute Value of Lambda Error",
             "Lambda Error over Number of Sites")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR,"lambda_error_by_sites.png"), dpi=150)
    plt.show()


def plot_p_error():
    est_p = _series("est_p")
    _, ax = plt.subplots(figsize=(8, 5))
    for method, color in zip(METHODS, COLORS):
        errors = [abs(e - TRUE_P) for e in est_p[method]]
        ax.plot(SITES, errors, marker="o", color=color, label=method)
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    _base_ax(ax, "Number of Sites", "Absolute Value of Detection Probability Error",
             "Error of Detection Probability by Number of Sites")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR,"p_error_by_sites.png"), dpi=150)
    plt.show()


def main():
    plot_total_N()
    plot_n_mae()
    plot_lambda_error()
    plot_p_error()


if __name__ == "__main__":
    main()
