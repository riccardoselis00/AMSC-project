#!/usr/bin/env python3
"""
Create grouped bar plots for MPI speedup of AS and AS2.

Input CSV must have columns:
    n,prec,nprocs,iters,residual,time_setup,time_solve,total_time

Definitions
-----------
Baseline (for *both* plots):
    t_base(n) = time_solve for prec = 'as' and nprocs = 1

Speedup for a given (n, prec, nprocs):
    speedup = t_base(n) / time_solve(n, prec, nprocs)

Plots
-----
1) AS: bars for prec='as', nprocs in {2,4,8,16}
2) AS2: bars for prec='as2', nprocs in {2,4,8,16}

x-axis: problem size n
y-axis: speedup vs AS (1 proc, no coarse)
"""

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_mpi_data(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    required = {"n", "prec", "nprocs", "time_solve"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    df["n"] = df["n"].astype(int)
    df["prec"] = df["prec"].astype(str)
    df["nprocs"] = df["nprocs"].astype(int)
    df["time_solve"] = df["time_solve"].astype(float)

    return df


# ---------------------------------------------------------------------
# Speedup computation
# ---------------------------------------------------------------------

def compute_speedup_vs_as1(
    df: pd.DataFrame,
    prec_target: str,
    procs: Iterable[int],
) -> pd.DataFrame:
    """
    For a target preconditioner (as or as2), compute speedup vs AS@1proc.

    Returns a DataFrame with columns:
        n, prec, nprocs, time_solve, t_base, speedup
    filtered to nprocs in 'procs'.
    """
    # baseline: AS with nprocs=1
    base = (
        df[(df["prec"] == "as") & (df["nprocs"] == 1)]
        .loc[:, ["n", "time_solve"]]
        .rename(columns={"time_solve": "t_base"})
    )

    # target prec
    dfp = df[df["prec"] == prec_target].copy()
    dfp = dfp.merge(base, on="n", how="inner")

    dfp = dfp[dfp["nprocs"].isin(procs)].copy()
    dfp["speedup"] = dfp["t_base"] / dfp["time_solve"]

    return dfp


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_grouped_bars(
    df_speed: pd.DataFrame,
    procs: Iterable[int],
    title: str,
    out_path: str | Path,
    ymax: float | None = None,
) -> None:
    """
    Grouped bar plot:
        x-axis : problem sizes n
        groups : one per n
        bars   : one per nprocs in 'procs'
    """
    procs = list(procs)
    ns = sorted(df_speed["n"].unique())
    x = np.arange(len(ns))

    width = 0.15  # bar width

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, p in enumerate(procs):
        g = df_speed[df_speed["nprocs"] == p]
        # ensure we have values ordered by ns
        g = g.set_index("n").reindex(ns)
        vals = g["speedup"].values

        ax.bar(
            x + i * width,
            vals,
            width,
            label=f"nprocs={p}",
        )

    # Center xticks under groups
    ax.set_xticks(x + width * (len(procs) - 1) / 2)
    ax.set_xticklabels([str(n) for n in ns])

    ax.set_xlabel("Problem size n")
    ax.set_xscale("log", base=2)
    ax.set_ylabel("Speedup vs AS (1 proc, no coarse)")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    if ymax is not None:
        ax.set_ylim(0, ymax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(
    csv_path: str = "data/output/csv/results_mpi_as_scaling.csv",
    out_dir: str = "data/output/images/speedup_mpi",
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_mpi_data(csv_path)

    # procs for bars (baseline p=1 is implicit)
    procs = [2, 4, 8, 16]

    # speedup tables
    df_as = compute_speedup_vs_as1(df, prec_target="as",  procs=procs)
    df_as2 = compute_speedup_vs_as1(df, prec_target="as2", procs=procs)

    # shared ymax for visual comparability
    max_speed = max(df_as["speedup"].max(), df_as2["speedup"].max())
    ymax = max_speed * 1.10

    # Plot for AS
    plot_grouped_bars(
        df_as,
        procs,
        title="AS speedup vs AS (1 proc, no coarse)",
        out_path=out_dir / "speedup_as_grouped.png",
        ymax=ymax,
    )

    # Plot for AS2
    plot_grouped_bars(
        df_as2,
        procs,
        title="AS2 speedup vs AS (1 proc, no coarse)",
        out_path=out_dir / "speedup_as2_grouped.png",
        ymax=ymax,
    )


if __name__ == "__main__":
    main()
