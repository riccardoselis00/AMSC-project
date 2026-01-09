import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator

# Load data
path = "results_mpi_as_scaling_blocks.csv"
df = pd.read_csv(path)

for col in ["n","nprocs","block_size","iters"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["prec"] = df["prec"].astype(str).str.lower().str.strip()

ns = [40000, 80000, 160000, 320000]
n_labels = {40000:"40k", 80000:"80k", 160000:"160k", 320000:"320k"}
grid_blocks = [1, 4, 16, 32]
procs_x = [1, 2, 4, 8, 16]

df_f = df[df["n"].isin(ns) & df["block_size"].isin(grid_blocks) & df["nprocs"].isin(procs_x) & df["prec"].isin(["as","as2"])].copy()

nrows, ncols = len(ns), len(grid_blocks)
fig_w, fig_h = 10.0, 13.2
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), sharey="row")

bar_width = 0.36
x = np.arange(len(procs_x))
label_fs, tick_fs, hdr_fs, legend_fs = 8, 7, 9, 9

# Precompute per-row max iterations to enforce visible y-scale
row_max = {}
for n in ns:
    m = df_f[df_f["n"] == n]["iters"].max()
    row_max[n] = float(m) if pd.notna(m) else 1.0

for i, n in enumerate(ns):
    for j, b in enumerate(grid_blocks):
        ax = axes[i, j]
        sub = df_f[(df_f["n"]==n) & (df_f["block_size"]==b)]

        it_as, it_as2 = [], []
        for p in procs_x:
            r_as = sub[(sub["prec"]=="as") & (sub["nprocs"]==p)]
            r_as2 = sub[(sub["prec"]=="as2") & (sub["nprocs"]==p)]
            it_as.append(float(r_as["iters"].iloc[0]) if len(r_as) else np.nan)
            it_as2.append(float(r_as2["iters"].iloc[0]) if len(r_as2) else np.nan)

        it_as = np.array(it_as, dtype=float)
        it_as2 = np.array(it_as2, dtype=float)

        ax.bar(x - bar_width/2, it_as, width=bar_width, label="AS" if (i==0 and j==0) else None)
        ax.bar(x + bar_width/2, it_as2, width=bar_width, label="AS2" if (i==0 and j==0) else None)

        # X ticks everywhere, on TOP only
        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in procs_x], fontsize=tick_fs)
        ax.xaxis.set_ticks_position('top')
        ax.tick_params(axis='x', labeltop=True, labelbottom=False, top=True, bottom=False, pad=2)
        ax.xaxis.set_label_position('top')
        #ax.set_xlabel("nprocs", fontsize=label_fs, labelpad=6)

        # Y ticks: only on left subplot of each row, but ensure they are ON and visible
        if j == 0:
            ax.tick_params(axis='y', left=True, labelleft=True, labelsize=tick_fs)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            sf = ScalarFormatter(useOffset=False)
            sf.set_scientific(False)
            ax.yaxis.set_major_formatter(sf)
            #ax.set_ylabel("Iterations", fontsize=label_fs)
        else:
            ax.tick_params(axis='y', left=False, labelleft=False)

        # Force y-limits so scale appears and includes max iterations
        ymax = row_max[n]
        ax.set_ylim(0, ymax * 1.05 if ymax > 0 else 1.0)

        ax.grid(axis="y", alpha=0.20)
        ax.set_xlim(-0.6, len(procs_x)-0.4)

        if np.all(np.isnan(it_as)) and np.all(np.isnan(it_as2)):
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=8)

# Column headers: nblocks
for j, b in enumerate(grid_blocks):
    axes[0, j].annotate(f"nblocks={b}", xy=(0.5, 1.22), xycoords="axes fraction",
                        ha="center", va="bottom", fontsize=hdr_fs)

# Row headers: n labels on the left of each row
for i, n in enumerate(ns):
    axes[i, 0].annotate(f"n={n_labels.get(n, str(n))}", xy=(-0.52, 0.5), xycoords="axes fraction",
                        ha="left", va="center", fontsize=hdr_fs, rotation=0)

# Global legend centered top, no global title
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=legend_fs, bbox_to_anchor=(0.5, 0.985))

fig.subplots_adjust(left=0.10, right=0.990, bottom=0.04, top=0.89, wspace=0.25, hspace=0.48)

out_dir = "as_as2_blocks_plots"
os.makedirs(out_dir, exist_ok=True)
png_path = os.path.join(out_dir, "iters_grid_top_xscale_left_yscale_WITH_ticks.png")
pdf_path = os.path.join(out_dir, "iters_grid_top_xscale_left_yscale_WITH_ticks.pdf")
fig.savefig(png_path, dpi=300)
fig.savefig(pdf_path)
plt.close(fig)

png_path, pdf_path
