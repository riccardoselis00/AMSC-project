#!/usr/bin/env python3
"""
plot_physical_problem.py

Reads the dumped files from:
  dd_project/data/output/plot-solutions/

File naming expected (from your C++ code):
  d{d}_Nx{Nx}_A.mtx
  d{d}_Nx{Nx}_b.txt
  d{d}_Nx{Nx}_x.txt
  d{d}_Nx{Nx}_grid.txt

Plots (for dim = 1,2,3):
  - Matrix sparsity pattern
  - Solution on the grid (1D curve, 2D heatmap, 3D mid-z slice)
  - Optional residual plots

Dependencies:
  numpy, matplotlib, scipy

Usage examples:
  python3 plot_physical_problem.py --dim all --Nx 20 --save
  python3 plot_physical_problem.py --dim 2 --Nx 50
  python3 plot_physical_problem.py --root /path/to/dd_project --dim all --Nx 20 --save
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import mmread


def _round_for_unique(x, nd=12):
    return np.round(x.astype(np.float64), nd)


def default_base_dir(root: str) -> str:
    # root = dd_project by default
    return os.path.join(root, "data", "output", "plot-solutions")


def load_case(base_dir: str, d: int, Nx: int):
    prefix = os.path.join(base_dir, f"d{d}_Nx{Nx}")
    A_path = prefix + "_A.mtx"
    b_path = prefix + "_b.txt"
    x_path = prefix + "_x.txt"
    g_path = prefix + "_grid.txt"

    A = None
    b = x = grid = None

    if os.path.exists(A_path):
        A = mmread(A_path).tocsr()
    else:
        print(f"[dim={d}] Missing matrix file: {A_path}")

    if os.path.exists(b_path):
        b = np.loadtxt(b_path)
    else:
        print(f"[dim={d}] Missing rhs file: {b_path}")

    if os.path.exists(x_path):
        x = np.loadtxt(x_path)
    else:
        print(f"[dim={d}] Missing solution file: {x_path}")

    if os.path.exists(g_path):
        grid = np.loadtxt(g_path)
    else:
        print(f"[dim={d}] Missing grid file: {g_path}")

    return A, b, x, grid


def plot_matrix_spy(A, d: int, Nx: int, save: bool, outdir: str, max_spy_n: int = 20000):
    if A is None:
        return

    n = A.shape[0]
    nnz = A.nnz
    print(f"[dim={d}] A: shape={A.shape}, nnz={nnz}")

    tag = f"d{d}_Nx{Nx}"

    if n <= max_spy_n:
        plt.figure()
        plt.spy(A, markersize=1)
        plt.title(f"Sparsity pattern A ({tag})")
        if save:
            path = os.path.join(outdir, f"spy_A_{tag}.png")
            plt.savefig(path, dpi=180, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
    else:
        m = max_spy_n
        Ablk = A[:m, :m]
        plt.figure()
        plt.spy(Ablk, markersize=1)
        plt.title(f"Sparsity A[:{m},:{m}] ({tag}, full n={n})")
        if save:
            path = os.path.join(outdir, f"spy_A_{tag}_topleft_{m}.png")
            plt.savefig(path, dpi=180, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def plot_grid_solution(d: int, Nx: int, grid, save: bool, outdir: str):
    if grid is None:
        return

    tag = f"d{d}_Nx{Nx}"

    if d == 1:
        xcoord = grid[:, 0]
        u = grid[:, 1]
        plt.figure()
        plt.plot(xcoord, u, marker="o")
        plt.xlabel("x")
        plt.ylabel("u(x)")
        plt.title(f"1D solution ({tag})")
        if save:
            plt.savefig(os.path.join(outdir, f"grid_{tag}_solution.png"), dpi=180, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    elif d == 2:
        X = grid[:, 0]
        Y = grid[:, 1]
        U = grid[:, 2]

        ux = np.unique(_round_for_unique(X))
        uy = np.unique(_round_for_unique(Y))
        nx = len(ux)
        ny = len(uy)
        print(f"[dim=2] inferred grid: Nx={nx}, Ny={ny}, points={len(U)}")

        # Your writer loops j then i (i fastest) => reshape (ny,nx) if consistent
        if nx * ny == len(U):
            U2 = U.reshape((ny, nx))
        else:
            order = np.lexsort((X, Y))
            U2 = U[order].reshape((ny, nx))

        extent = [ux.min(), ux.max(), uy.min(), uy.max()]
        plt.figure()
        plt.imshow(U2, origin="lower", aspect="auto", extent=extent)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"2D solution heatmap ({tag})")
        plt.colorbar(label="u")
        if save:
            plt.savefig(os.path.join(outdir, f"grid_{tag}_heatmap.png"), dpi=180, bbox_inches="tight", colormap="turbo")
            plt.close()
        else:
            plt.show()

    elif d == 3:
        X = grid[:, 0]
        Y = grid[:, 1]
        Z = grid[:, 2]
        U = grid[:, 3]

        ux = np.unique(_round_for_unique(X))
        uy = np.unique(_round_for_unique(Y))
        uz = np.unique(_round_for_unique(Z))
        nx, ny, nz = len(ux), len(uy), len(uz)
        print(f"[dim=3] inferred grid: Nx={nx}, Ny={ny}, Nz={nz}, points={len(U)}")

        if nx * ny * nz == len(U):
            U3 = U.reshape((nz, ny, nx))
        else:
            order = np.lexsort((X, Y, Z))
            U3 = U[order].reshape((nz, ny, nx))

        k0 = nz // 2
        U_slice = U3[k0, :, :]
        extent = [ux.min(), ux.max(), uy.min(), uy.max()]

        plt.figure()
        plt.imshow(U_slice, origin="lower", aspect="auto", extent=extent)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"3D mid-slice (k={k0}) ({tag})")
        plt.colorbar(label="u")
        if save:
            plt.savefig(os.path.join(outdir, f"grid_{tag}_slice_k{k0}.png"), dpi=180, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def residual_check(A, b, x, d: int, Nx: int, save: bool, outdir: str):
    if A is None or b is None or x is None:
        return

    tag = f"d{d}_Nx{Nx}"

    if A.shape[0] != x.shape[0] or A.shape[0] != b.shape[0]:
        print(f"[dim={d}] WARNING: size mismatch A:{A.shape}, x:{x.shape}, b:{b.shape}")
        return

    r = A @ x - b
    bn = np.linalg.norm(b)
    rn = np.linalg.norm(r)
    rel = rn / (bn if bn > 0 else 1.0)
    print(f"[dim={d}] residual: ||Ax-b||={rn:.3e}, rel={rel:.3e}")

    plt.figure()
    plt.semilogy(np.abs(r) + 1e-300)
    plt.xlabel("i")
    plt.ylabel("|r_i|")
    plt.title(f"Residual entries ({tag}), rel={rel:.3e}")
    if save:
        plt.savefig(os.path.join(outdir, f"residual_{tag}.png"), dpi=180, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def run(dim_choice: str, base_dir: str, Nx: int, save: bool, outdir: str, do_residual: bool):
    os.makedirs(outdir, exist_ok=True)

    dims = [1, 2, 3] if dim_choice == "all" else [int(dim_choice)]

    for d in dims:
        print(f"\n=== Plotting dim={d}, Nx={Nx} ===")
        A, b, x, grid = load_case(base_dir, d, Nx)

        plot_matrix_spy(A, d, Nx, save, outdir)
        plot_grid_solution(d, Nx, grid, save, outdir)

        if do_residual:
            residual_check(A, b, x, d, Nx, save, outdir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", default="all", choices=["all", "1", "2", "3"],
                    help="Which dimension to plot")
    ap.add_argument("--Nx", type=int, default=20, help="Nx used in file names (d*_Nx{Nx}_...)")

    # root = dd_project directory; default assumes you run from dd_project/build or dd_project
    ap.add_argument("--root", default="..",
                    help="Path to dd_project (default: '..' which works from build/)")
    ap.add_argument("--dir", default=None,
                    help="Explicit directory containing plot-solutions files (overrides --root)")

    ap.add_argument("--save", action="store_true", help="Save PNGs instead of showing interactively")
    ap.add_argument("--out", default="plots", help="Output directory for PNGs (when --save)")
    ap.add_argument("--residual", action="store_true", help="Also compute/plot residual Ax-b")

    args = ap.parse_args()

    base_dir = args.dir if args.dir is not None else default_base_dir(args.root)
    if not os.path.isdir(base_dir):
        raise SystemExit(f"Base directory not found: {base_dir}")

    run(args.dim, base_dir, args.Nx, args.save, args.out, args.residual)


if __name__ == "__main__":
    main()
