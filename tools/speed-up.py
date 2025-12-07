import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import sys
csv_file = sys.argv[1]

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
    plt.bar(x + i * bar_width, pivot[b], width=bar_width, label=f"AS block={b}")

plt.xticks(x + 1.5 * bar_width, pivot.index)
plt.xlabel("Problem size n")
plt.ylabel("Speedup vs Identity")
# plt.title("Additive Schwarz Speedup over Identity Preconditioner")
plt.legend()
# plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
