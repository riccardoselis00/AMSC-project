#!/usr/bin/env python3
"""
Utility functions to post-process PCG + preconditioner timing results
and generate plots.

Expected CSV columns:

    preconditioner   : string label (e.g., "Pr1", "Pr2", ...)
    dim              : matrix dimension n (for an n x n system)
    apply_prec_time  : time spent per solve in the preconditioner apply (seconds)
    iterations       : number of PCG iterations
    time_per_iter    : average time per iteration (seconds)

Each row corresponds to one (preconditioner, dim) pair.
"""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------

def load_data(csv_path: str | Path) -> pd.DataFrame:
    """
    Load the CSV file and enforce basic types.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with typed columns.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    expected = {"preconditioner", "dim", "apply_prec_time", "iterations", "time_per_iter"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    df["preconditioner"] = df["preconditioner"].astype(str)
    df["dim"] = df["dim"].astype(int)
    df["apply_prec_time"] = df["apply_prec_time"].astype(float)
    df["iterations"] = df["iterations"].astype(int)
    df["time_per_iter"] = df["time_per_iter"].astype(float)

    return df


def add_total_solve_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'total_solve_time' column = iterations * time_per_iter.

    Returns a copy, does not modify the original DataFrame in-place.
    """
    df2 = df.copy()
    df2["total_solve_time"] = df2["iterations"] * df2["time_per_iter"]
    return df2


def pivot_metric(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Build a table with:
        rows   = preconditioner
        cols   = matrix dimension
        values = chosen metric (value_col).

    Good for debugging and exporting, but plotting does not require this.
    """
    table = df.pivot(index="preconditioner", columns="dim", values=value_col)
    table = table.reindex(sorted(table.columns), axis=1)
    return table


def compute_all_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Convenience: given a DataFrame (already loaded), compute pivot tables
    for key metrics.

    Returns
    -------
    tables : dict[str, DataFrame]
        Keys:
            - "apply_prec_time"
            - "iterations"
            - "time_per_iter"
            - "total_solve_time" (if present in df)
    """
    metrics = ["apply_prec_time", "iterations", "time_per_iter"]
    if "total_solve_time" in df.columns:
        metrics.append("total_solve_time")

    tables: Dict[str, pd.DataFrame] = {}
    for metric in metrics:
        tables[metric] = pivot_metric(df, metric)

    return tables


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
    Plot a metric vs matrix dimension, one curve per preconditioner.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns:
        'preconditioner', 'dim', and the chosen metric.
    metric : str
        Name of the metric column to plot
        (e.g. 'iterations', 'apply_prec_time', 'time_per_iter',
        or 'total_solve_time' if added).
    logx : bool, default True
        Use logarithmic scale on x-axis (dimensions n, 2n, 4n, ...).
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

    # Create axes if needed
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Sort dimensions so curves are nice and monotone on x-axis
    # (n, 2n, 4n, 8n, ...)
    # We do this per preconditioner to avoid problems if some are missing.
    for pr, group in df.groupby("preconditioner"):
        group_sorted = group.sort_values("dim")
        ax.plot(
            group_sorted["dim"],
            group_sorted[metric],
            marker="o",
            linestyle="-",
            label=pr,
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
