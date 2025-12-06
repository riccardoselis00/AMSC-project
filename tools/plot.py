#!/usr/bin/env python3
"""
Utility functions to post-process PCG + preconditioner timing results
and generate plots.

Expected CSV columns (like your current output):

    n,prec,iters,residual,time_setup,time_solve, total time

Notes:
- Column names are stripped, so ' total time' becomes 'total time'.
- We then rename 'total time' -> 'total_time' internally.
- We derive: time_per_iter = time_solve / iters.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------

def load_data(csv_path: str | Path) -> pd.DataFrame:
    """
    Load the CSV file produced by the C++ benchmark and normalize columns.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with typed and normalized columns:
        n, prec, iters, residual, time_setup, time_solve, total_time, time_per_iter
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # Strip whitespace from column names (e.g. " total time" -> "total time")
    df.columns = [c.strip() for c in df.columns]

    # Rename "total time" -> "total_time" if present
    if "total time" in df.columns:
        df = df.rename(columns={"total time": "total_time"})

    # Basic expected columns
    expected = {"n", "prec", "iters", "residual", "time_setup", "time_solve"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    # Types
    df["n"] = df["n"].astype(int)
    df["prec"] = df["prec"].astype(str)
    df["iters"] = df["iters"].astype(int)
    df["residual"] = df["residual"].astype(float)
    df["time_setup"] = df["time_setup"].astype(float)
    df["time_solve"] = df["time_solve"].astype(float)

    # total_time: if present, cast; otherwise compute as setup + solve
    if "total_time" in df.columns:
        df["total_time"] = df["total_time"].astype(float)
    else:
        df["total_time"] = df["time_setup"] + df["time_solve"]

    # Derived metric: average time per iteration
    df["time_per_iter"] = df["time_solve"] / df["iters"].where(df["iters"] != 0, 1)

    return df


def add_total_solve_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    For backward compatibility with your notebook.

    Adds a 'total_solve_time' column. Here we set:

        total_solve_time = total_time

    since `total_time` is already in the CSV (or computed in load_data).
    """
    df2 = df.copy()
    if "total_time" not in df2.columns:
        df2["total_time"] = df2["time_setup"] + df2["time_solve"]
    df2["total_solve_time"] = df2["total_time"]
    return df2


# ---------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------

def plot_metric(
    df: pd.DataFrame,
    metric: str,
    *,
    logx: bool = True,
    logy: bool = False,
    ax: plt.Axes | None = None,
    title: str | None = None,
    metric_label: str | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a metric vs matrix dimension n, one curve per preconditioner (prec).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns:
        'prec', 'n', and the chosen metric.
    metric : str
        Name of the metric column to plot
        (e.g. 'iters', 'time_setup', 'time_solve', 'total_time', 'time_per_iter').
    logx : bool, default True
        Use logarithmic scale on x-axis.
    logy : bool, default False
        Use logarithmic scale on y-axis.
    ax : matplotlib.axes.Axes or None
        If provided, plot into this axes. If None, create a new figure.
    title : str or None
        Optional title.
    metric_label : str or None
        Label for y-axis. If None, use the metric name.

    Returns
    -------
    fig, ax : (Figure, Axes)
        The matplotlib figure and axes objects.
    """
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in DataFrame columns.")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    for prec, group in df.groupby("prec"):
        group_sorted = group.sort_values("n")
        ax.plot(
            group_sorted["n"],
            group_sorted[metric],
            marker="o",
            linestyle="-",
            label=prec,
        )

    if logx:
        ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")

    ax.set_xlabel("Matrix dimension n")
    ax.set_ylabel(metric_label if metric_label is not None else metric)

    if title is not None:
        ax.set_title(title)

    ax.legend(title="Preconditioner")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    fig.tight_layout()
    return fig, ax
