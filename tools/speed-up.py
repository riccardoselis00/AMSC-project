import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import sys
csv_file = sys.argv[1]

# ------------------------- TEXT / FONT SIZES (EDIT HERE) -------------------------
plt.rcParams.update({
    "font.size": 24,          # base text
    "axes.titlesize": 20,     # subplot titles
    "axes.labelsize": 20,     # x/y labels
    "xtick.labelsize": 20,    # x ticks
    "ytick.labelsize": 20,    # y ticks
    "legend.fontsize": 20,
})
# -------------------------------------------------------------------------------

# -----------------------------------------------------
# Load and clean CSV
# -----------------------------------------------------
df = pd.read_csv(csv_file)

# n must be integer
df["n"] = df["n"].astype(str).str.strip().astype(int)

# -----------------------------------------------------
# Extract identity baseline times
# -----------------------------------------------------
identity = (
    df[df["prec"] == "identity"][["n", "total_time"]]
    .set_index("n")["total_time"]
    .to_dict()
)

print("Identity rows loaded:", identity)

# -----------------------------------------------------
# Compute AS speedups
# -----------------------------------------------------
as_data = df[df["prec"] == "as"].copy()

def compute_speedup(row):
    n = int(row["n"])
    return identity[n] / row["total_time"]

as_data["speedup"] = as_data.apply(compute_speedup, axis=1)

# -----------------------------------------------------
# Pivot for plotting
# -----------------------------------------------------
pivot = as_data.pivot(index="n", columns="block_size", values="speedup")
pivot = pivot[[4, 8, 16, 32]]  # enforce order

print("\nSpeedup table:")
print(pivot)

# -----------------------------------------------------
# Bar Plot
# -----------------------------------------------------
plt.figure(figsize=(12, 8))

x = np.arange(len(pivot.index))
bar_width = 0.15

for i, b in enumerate(pivot.columns):
    plt.bar(x + i * bar_width, pivot[b], width=bar_width, label=f"blocks={b}")

# Format ticks as 40k, 80k, ...
def fmt_k(n: int) -> str:
    n = int(n)
    if n % 1000 == 0:
        return f"{n//1000}k"
    return f"{n/1000:.1f}k"  # fallback (e.g., 12500 -> 12.5k)

plt.xticks(x + 1.5 * bar_width, [fmt_k(n) for n in pivot.index])
plt.xlabel("N")
plt.ylabel("Speedup")

#plt.legend()
plt.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=len(pivot.columns),
    frameon=False
)

plt.tight_layout(rect=[0, 0, 0, 0.92])
plt.show()

# plt.tight_layout()
# plt.show()
