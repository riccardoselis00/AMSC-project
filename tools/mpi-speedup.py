#!/usr/bin/env python3
"""
Plot MPI speedup for AS and AS2 preconditioners.

- Input CSV must have columns:
    n,prec,nprocs,iters,residual,time_setup,time_solve,total_time

- Baseline: AS with nprocs = 1 (per each n)
  speedup(n, prec, p) = time_solve(n, AS, p=1) / time_solve(n, prec, p)

- For each problem size n, we create a figure with:
    x-axis: nprocs
    y-axis: speedup
    curves: AS and AS2
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


def load_mpi_data(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # Basic sanity: required columns
    required = {"n", "prec", "nprocs", "time_solve"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    # Enforce dtypes
    df["n"] = df["n"].astype(int)
    df["prec"] = df["prec"].astype(str)
    df["nprocs"] = df["nprocs"].astype(int)
    df["time_solve"] = df["time_solve"].astype(float)

    # Optional columns
    if "time_setup" in df.columns:
        df["time_setup"] = df["time_setup"].astype(float)
    if "total_time" in df.columns:
        df["total_time"] = df["total_time"].astype(float)

    return df


def compute_speedup_vs_as1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'speedup' column:
        speedup = time_solve(as, nprocs=1) / time_solve(prec, nprocs=p)

    We do this per (n, prec, nprocs).
    """
    # Baseline: AS with nprocs=1
    base = (
        df[(df["prec"] == "as") & (df["nprocs"] == 1)]
        .loc[:, ["n", "time_solve"]]
        .rename(columns={"time_solve": "t_as_np1"})
    )

    # Join baseline onto all rows by n
    df2 = df.merge(base, on="n", how="left")

    # Drop rows where we have no baseline (just in case)
    df2 = df2.dropna(subset=["t_as_np1"])

    # Compute speedup
    df2["speedup"] = df2["t_as_np1"] / df2["time_solve"]

    return df2


def plot_speedup_per_n(
    df_speed: pd.DataFrame,
    out_dir: str | Path = "plots",
    x_log2: bool = True,
    show: bool = False,
) -> None:
    """
    For each problem size n, create a PNG with speedup vs nprocs for prec ∈ {as, as2}.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Only keep as/as2
    dfp = df_speed[df_speed["prec"].isin(["as", "as2"])].copy()

    for n, group in dfp.groupby("n"):
        fig, ax = plt.subplots()

        # One curve per preconditioner
        for prec, gprec in group.groupby("prec"):
            gprec = gprec.sort_values("nprocs")
            label = "AS (1 level)" if prec == "as" else "AS2 (2 levels)"

            ax.plot(
                gprec["nprocs"],
                gprec["speedup"],
                marker="o",
                linestyle="-",
                label=label,
            )

        if x_log2:
            ax.set_xscale("log", base=2)
            ax.set_yscale("log", base=2)

        ax.set_xlabel("Number of MPI processes")
        ax.set_ylabel("Speedup vs AS, nprocs=1")
        ax.set_title(f"Speedup vs #procs, n = {n}")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.legend()

        fig.tight_layout()
        out_path = out_dir / f"speedup_n{n}.png"
        fig.savefig(out_path, dpi=200)

        if show:
            plt.show()
        else:
            plt.close(fig)


def main(
    csv_path: str = "data/output/csv/results_mpi_as_scaling.csv",
    out_dir: str = "data/output/images/speedup_mpi",
) -> None:
    df = load_mpi_data(csv_path)
    df_speed = compute_speedup_vs_as1(df)
    plot_speedup_per_n(df_speed, out_dir=out_dir, x_log2=True, show=False)


if __name__ == "__main__":
    main()
