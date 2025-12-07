"""
Create ONE image with two subplots:
- LEFT: Strong scaling speedup
- RIGHT: Weak scaling efficiency

Takes **two CSV inputs**:
    python plot_scaling.py strong.csv weak.csv

CSV format must contain:
n,prec,nprocs,iters,residual,time_setup,time_solve,total_time
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df["n"] = df["n"].astype(int)
    df["nprocs"] = df["nprocs"].astype(int)
    df["time_solve"] = df["time_solve"].astype(float)
    return df

def compute_strong_scaling(df):
    strong = {}

    for N, group in df.groupby("n"):
        base = group[group["nprocs"] == 1]["time_solve"].values
        if len(base) == 0:
            continue

        T1 = base[0]
        g = group.sort_values("nprocs")

        strong[N] = {
            "procs": g["nprocs"].values,
            "speedup": T1 / g["time_solve"].values,
        }

    return strong

def compute_weak_scaling(df):
    weak = {}

    base = df[df["nprocs"] == 1]["time_solve"].values
    if len(base) == 0:
        print("WARNING: weak CSV missing nprocs=1 baseline")
        return weak

    T1 = base[0]

    for p, group in df.groupby("nprocs"):
        Tp = group["time_solve"].mean()
        weak[p] = T1 / Tp

    return weak


def plot_two_panels(strong, weak, out_path):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    for N, data in strong.items():
        ax1.plot(
            data["procs"],
            data["speedup"],
            marker="o",
            linestyle="-",
            label=f"N = {N}",
        )

    ax1.set_title("Strong Scaling (Speedup vs #procs)")
    ax1.set_xlabel("# MPI Processes")
    ax1.set_xscale("log", base=2)
    ax1.set_ylabel("Speedup = T1 / Tp")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()

    procs = sorted(weak.keys())
    eff = [weak[p] for p in procs]

    ax2.plot(procs, eff, marker="o", linestyle="-", color="purple")

    ax2.set_title("Weak Scaling (Efficiency vs #procs)")
    ax2.set_xlabel("# MPI Processes")
    ax2.set_xscale("log", base=2)
    ax2.set_ylabel("Efficiency = T1 / Tp")
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    print(f"[OK] Saved: {out_path}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python plot_scaling.py strong.csv weak.csv")
        sys.exit(1)

    strong_csv = sys.argv[1]
    weak_csv = sys.argv[2]

    df_strong = load_data(strong_csv)
    df_weak = load_data(weak_csv)

    strong = compute_strong_scaling(df_strong)
    weak = compute_weak_scaling(df_weak)

    plot_two_panels(strong, weak, "data/output/images/scaling_summary.png")


if __name__ == "__main__":
    main()
