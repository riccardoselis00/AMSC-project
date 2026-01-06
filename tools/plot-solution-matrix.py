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

Outputs:
  - If --dim all:
      * solutions_row_Nx{Nx}.png  (1D + 2D + 3D) with ONE shared color scale
        -> colorbar is placed on the LEFT of the 1D plot
        -> 1D y-scale is ON and matches the shared scale (blue=min, red=max)
      * matrices_row_Nx{Nx}.png   (spy d=1 + d=2 + d=3)
  - If --dim 1/2/3:
      * separate images per dimension (plus spy)

Color scale control (IMPORTANT):
  --scale-mode 3d    : vmin/vmax taken from 3D solution ONLY (default) => red=max(3D), blue=min(3D)
  --scale-mode all   : vmin/vmax taken from 1D+2D+3D combined
  --scale-mode manual --vmin A --vmax B : manually force color meaning

Text size control (for report visibility):
  Change these rcParams below (font.size, axes.labelsize, xtick.labelsize, etc.)

Dependencies:
  numpy, matplotlib, scipy
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import mmread
from scipy.ndimage import gaussian_filter

# ------------------------- TEXT / FONT SIZES (EDIT HERE) -------------------------
plt.rcParams.update({
    "font.size": 14,          # base text
    "axes.titlesize": 15,     # subplot titles
    "axes.labelsize": 13,     # x/y labels
    "xtick.labelsize": 12,    # x ticks
    "ytick.labelsize": 12,    # y ticks
    "legend.fontsize": 12,
})
# -------------------------------------------------------------------------------


def _round_for_unique(x, nd=12):
    return np.round(x.astype(np.float64), nd)


def default_base_dir(root: str) -> str:
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


# ------------------------- Matrix plotting (spy) -------------------------

def plot_matrix_spy_ax(ax, A, d: int, Nx: int, max_spy_n: int = 20000):
    ax.set_title(f"Spy A (d={d}, Nx={Nx})")
    ax.set_xlabel("j")
    ax.set_ylabel("i")

    if A is None:
        ax.text(0.5, 0.5, "Missing A", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    n = A.shape[0]
    nnz = A.nnz
    print(f"[dim={d}] A: shape={A.shape}, nnz={nnz}")

    if n <= max_spy_n:
        ax.spy(A, markersize=0.7)
    else:
        m = max_spy_n
        ax.spy(A[:m, :m], markersize=0.7)
        ax.set_title(f"Spy A[:{m},:{m}] (d={d}, Nx={Nx})")


# ------------------------- 2D / 3D helpers -------------------------

def _infer_2d_grid(grid):
    X = grid[:, 0]
    Y = grid[:, 1]
    U = grid[:, 2]

    ux = np.unique(_round_for_unique(X))
    uy = np.unique(_round_for_unique(Y))

    order = np.lexsort((X, Y))  # primary Y, secondary X
    U2 = U[order].reshape((len(uy), len(ux)))
    return ux, uy, U2


def _infer_3d_grid(grid):
    X = grid[:, 0]
    Y = grid[:, 1]
    Z = grid[:, 2]
    U = grid[:, 3]

    ux = np.unique(_round_for_unique(X))
    uy = np.unique(_round_for_unique(Y))
    uz = np.unique(_round_for_unique(Z))

    order = np.lexsort((X, Y, Z))  # primary Z, then Y, then X
    U3 = U[order].reshape((len(uz), len(uy), len(ux)))
    return ux, uy, uz, U3


# ------------------------- 1D colored curve (WITH LEFT SCALE MATCHING NORM) -------------------------

def plot_solution_1d_ax(ax, grid, cmap, norm):
    ax.set_title("1D")

    if grid is None:
        ax.text(0.5, 0.5, "Missing grid",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    xcoord = grid[:, 0].astype(np.float64)
    u = grid[:, 1].astype(np.float64)

    pts = np.column_stack([xcoord, u])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    u_mid = 0.5 * (u[:-1] + u[1:])

    lc = LineCollection(segs, cmap=cmap, norm=norm)
    lc.set_array(u_mid)
    lc.set_linewidth(2.2)
    ax.add_collection(lc)

    ax.set_xlim(xcoord.min(), xcoord.max())

    # data-driven y-limits (correct, keep this)
    umin, umax = u.min(), u.max()
    pad = 0.05 * (umax - umin if umax > umin else 1.0)
    ax.set_ylim(umin - pad, umax + pad)

    ax.set_xlabel("x")

    # 🔴 THIS IS WHAT YOU WANT
    ax.set_yticks([])                 # remove y tick numbers
    ax.set_ylabel("")                 # remove y label text
    #ax.spines["left"].set_visible(False)  # optional: hide left axis line

    ax.grid(False)



def plot_solution_2d_ax(ax, grid, blur_sigma: float, cmap, norm):
    ax.set_title("2D")
    if grid is None:
        ax.text(0.5, 0.5, "Missing grid", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return None

    ux, uy, U2 = _infer_2d_grid(grid)
    print(f"[dim=2] inferred grid: Nx={len(ux)}, Ny={len(uy)}, points={U2.size}")

    U2p = gaussian_filter(U2, sigma=blur_sigma) if (blur_sigma and blur_sigma > 0) else U2
    extent = [ux.min(), ux.max(), uy.min(), uy.max()]

    im = ax.imshow(
        U2p,
        origin="lower",
        extent=extent,
        aspect="equal",
        interpolation="bilinear" if (blur_sigma and blur_sigma > 0) else "nearest",
        cmap=cmap,
        norm=norm,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(False)
    return im


# ------------------------- 3D cube + slices -------------------------

def _draw_cube_wireframe(ax, xmin, xmax, ymin, ymax, zmin, zmax, lw=1.0):
    P = np.array([
        [xmin, ymin, zmin],
        [xmax, ymin, zmin],
        [xmax, ymax, zmin],
        [xmin, ymax, zmin],
        [xmin, ymin, zmax],
        [xmax, ymin, zmax],
        [xmax, ymax, zmax],
        [xmin, ymax, zmax],
    ])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        ax.plot([P[i, 0], P[j, 0]],
                [P[i, 1], P[j, 1]],
                [P[i, 2], P[j, 2]], linewidth=lw)


def plot_solution_3d_cube_slices_ax(ax, grid, z_slices, blur_sigma: float, cmap, norm):
    ax.set_title("3D")
    if grid is None:
        ax.text2D(0.5, 0.5, "Missing grid", ha="center", va="center", transform=ax.transAxes)
        return

    ux, uy, uz, U3 = _infer_3d_grid(grid)
    nx, ny, nz = len(ux), len(uy), len(uz)
    print(f"[dim=3] inferred grid: Nx={nx}, Ny={ny}, Nz={nz}, points={U3.size}")

    xmin, xmax = float(ux.min()), float(ux.max())
    ymin, ymax = float(uy.min()), float(uy.max())
    zmin, zmax = float(uz.min()), float(uz.max())

    ax.view_init(elev=22, azim=-58)
    ax.grid(False)

    # Clean panes
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.set_alpha(0.0)
        except Exception:
            pass

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    _draw_cube_wireframe(ax, xmin, xmax, ymin, ymax, zmin, zmax, lw=1.0)

    XX, YY = np.meshgrid(ux, uy, indexing="xy")

    alpha = 0.92
    for zt in z_slices:
        k = int(np.argmin(np.abs(uz - zt)))
        z_actual = float(uz[k])

        Usl = U3[k, :, :]
        Uslp = gaussian_filter(Usl, sigma=blur_sigma) if (blur_sigma and blur_sigma > 0) else Usl

        # IMPORTANT: cmap(norm(...)) => blue=min, red=max (if cmap is not reversed)
        facecolors = cmap(norm(Uslp))
        facecolors[..., -1] = alpha

        ZZ = np.full_like(XX, z_actual, dtype=np.float64)

        ax.plot_surface(
            XX, YY, ZZ,
            facecolors=facecolors,
            rstride=1, cstride=1,
            linewidth=0.0,
            antialiased=False,
            shade=False,
        )


# ------------------------- Shared scale computation -------------------------

def _range_all_dims(grids):
    vals = []
    if grids.get(1) is not None:
        vals.append(grids[1][:, 1])
    if grids.get(2) is not None:
        vals.append(grids[2][:, 2])
    if grids.get(3) is not None:
        vals.append(grids[3][:, 3])
    if not vals:
        return 0.0, 1.0

    v = np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in vals])
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-12
    return vmin, vmax


def _range_3d_only(grids):
    g3 = grids.get(3)
    if g3 is None:
        return _range_all_dims(grids)

    u3 = np.asarray(g3[:, 3], dtype=np.float64).ravel()
    vmin = float(np.min(u3))
    vmax = float(np.max(u3))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-12
    return vmin, vmax


def compute_norm(grids, scale_mode: str, vmin_arg, vmax_arg):
    if scale_mode == "manual":
        if vmin_arg is None or vmax_arg is None:
            raise SystemExit("ERROR: --scale-mode manual requires BOTH --vmin and --vmax")
        vmin = float(vmin_arg)
        vmax = float(vmax_arg)
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-12
        return mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    if scale_mode == "all":
        vmin, vmax = _range_all_dims(grids)
        return mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # default: "3d"
    vmin, vmax = _range_3d_only(grids)
    return mpl.colors.Normalize(vmin=vmin, vmax=vmax)


# ------------------------- Combined figures -------------------------

def plot_all_solutions_row(base_dir: str, Nx: int, save: bool, outdir: str,
                           blur_sigma: float, z_slices, cmap_name: str,
                           scale_mode: str, vmin_arg, vmax_arg):

    # Load grids
    grids = {}
    for d in (1, 2, 3):
        _, _, _, g = load_case(base_dir, d, Nx)
        grids[d] = g

    cmap = mpl.cm.get_cmap(cmap_name)  # use "turbo" for blue->...->red (red=max)
    norm = compute_norm(grids, scale_mode, vmin_arg, vmax_arg)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    fig = plt.figure(figsize=(16, 4.8))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")

    plot_solution_1d_ax(ax1, grids[1], cmap=cmap, norm=norm)
    plot_solution_2d_ax(ax2, grids[2], blur_sigma=blur_sigma, cmap=cmap, norm=norm)
    plot_solution_3d_cube_slices_ax(ax3, grids[3], z_slices=z_slices, blur_sigma=blur_sigma, cmap=cmap, norm=norm)

    # COLORBAR ON THE LEFT OF THE 1D PLOT
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("left", size="4.5%", pad=0.35)

    cbar = fig.colorbar(sm, cax=cax)
    #cbar.set_label("u", fontsize=plt.rcParams["axes.labelsize"])
    cbar.ax.yaxis.set_label_position("left")
    cbar.ax.yaxis.tick_left()

    # Improve spacing (so left colorbar has room)
    fig.tight_layout()

    if save:
        path = os.path.join(outdir, f"solutions_row_Nx{Nx}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {path}")
    else:
        plt.show()


def plot_all_matrices_row(base_dir: str, Nx: int, save: bool, outdir: str, max_spy_n: int):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for idx, d in enumerate([1, 2, 3]):
        A, _, _, _ = load_case(base_dir, d, Nx)
        plot_matrix_spy_ax(axes[idx], A, d=d, Nx=Nx, max_spy_n=max_spy_n)

    fig.tight_layout()
    if save:
        path = os.path.join(outdir, f"matrices_row_Nx{Nx}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {path}")
    else:
        plt.show()


# ------------------------- Residual -------------------------

def residual_check(A, b, x, d: int, Nx: int, save: bool, outdir: str):
    if A is None or b is None or x is None:
        return
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
    plt.title(f"Residual entries (d={d}, Nx={Nx}), rel={rel:.3e}")
    plt.grid(False)
    if save:
        path = os.path.join(outdir, f"residual_d{d}_Nx{Nx}.png")
        plt.savefig(path, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"[saved] {path}")
    else:
        plt.show()


# ------------------------- Run -------------------------

def _parse_csv_floats(s: str):
    return [float(t.strip()) for t in s.split(",") if t.strip()]


def run(dim_choice: str, base_dir: str, Nx: int, save: bool, outdir: str,
        do_residual: bool, blur_sigma: float, z_slices, cmap_name: str, max_spy_n: int,
        scale_mode: str, vmin_arg, vmax_arg):

    os.makedirs(outdir, exist_ok=True)

    if dim_choice == "all":
        plot_all_solutions_row(
            base_dir, Nx, save, outdir,
            blur_sigma=blur_sigma, z_slices=z_slices, cmap_name=cmap_name,
            scale_mode=scale_mode, vmin_arg=vmin_arg, vmax_arg=vmax_arg
        )
        plot_all_matrices_row(base_dir, Nx, save, outdir, max_spy_n)

        if do_residual:
            for d in [1, 2, 3]:
                A, b, x, _ = load_case(base_dir, d, Nx)
                residual_check(A, b, x, d, Nx, save, outdir)
        return

    # Single-dim mode: keep simple per-plot colorbar on the right
    d = int(dim_choice)
    print(f"\n=== Plotting dim={d}, Nx={Nx} ===")
    A, b, x, grid = load_case(base_dir, d, Nx)

    # matrix
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plot_matrix_spy_ax(ax, A, d=d, Nx=Nx, max_spy_n=max_spy_n)
    fig.tight_layout()
    if save:
        path = os.path.join(outdir, f"spy_A_d{d}_Nx{Nx}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {path}")
    else:
        plt.show()

    # solution
    cmap = mpl.cm.get_cmap(cmap_name)

    if d == 1:
        if grid is None:
            print("[dim=1] Missing grid; skipping")
            return
        u = grid[:, 1].astype(np.float64)
        vmin, vmax = float(np.min(u)), float(np.max(u))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-12
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))
        plot_solution_1d_ax(ax, grid, cmap=cmap, norm=norm)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04).set_label("u")
        fig.tight_layout()
        if save:
            path = os.path.join(outdir, f"solution_d{d}_Nx{Nx}.png")
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"[saved] {path}")
        else:
            plt.show()

    elif d == 2:
        if grid is None:
            print("[dim=2] Missing grid; skipping")
            return
        _, _, U2 = _infer_2d_grid(grid)
        vmin, vmax = float(np.min(U2)), float(np.max(U2))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-12
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.5))
        im = plot_solution_2d_ax(ax, grid, blur_sigma=blur_sigma, cmap=cmap, norm=norm)
        if im is not None:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("u")
        fig.tight_layout()
        if save:
            path = os.path.join(outdir, f"solution_d{d}_Nx{Nx}.png")
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"[saved] {path}")
        else:
            plt.show()

    elif d == 3:
        if grid is None:
            print("[dim=3] Missing grid; skipping")
            return
        _, _, _, U3 = _infer_3d_grid(grid)
        vmin, vmax = float(np.min(U3)), float(np.max(U3))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-12
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        fig = plt.figure(figsize=(20.5, 20.2))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        plot_solution_3d_cube_slices_ax(ax, grid, z_slices=z_slices, blur_sigma=blur_sigma, cmap=cmap, norm=norm)
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04).set_label("u")
        fig.tight_layout()
        if save:
            path = os.path.join(outdir, f"solution_d{d}_Nx{Nx}_cube_slices.png")
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"[saved] {path}")
        else:
            plt.show()

    if do_residual:
        residual_check(A, b, x, d, Nx, save, outdir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", default="all", choices=["all", "1", "2", "3"])
    ap.add_argument("--Nx", type=int, default=20)

    ap.add_argument("--root", default="..")
    ap.add_argument("--dir", default=None)

    ap.add_argument("--save", action="store_true")
    ap.add_argument("--out", default="plots")
    ap.add_argument("--residual", action="store_true")

    ap.add_argument("--z-slices", default="0.1,0.5,0.9", help="Comma-separated z planes (nearest planes)")
    ap.add_argument("--blur-sigma", type=float, default=0.8, help="Gaussian blur sigma (0 disables)")
    ap.add_argument("--cmap", default="turbo", help="Use 'turbo' for blue(min)->red(max)")

    ap.add_argument("--max-spy-n", type=int, default=20000)

    # NEW: control color meaning
    ap.add_argument("--scale-mode", default="3d", choices=["3d", "all", "manual"],
                    help="Color scale source. 3d => red=max(3D), blue=min(3D).")
    ap.add_argument("--vmin", type=float, default=None, help="Manual vmin (only with --scale-mode manual)")
    ap.add_argument("--vmax", type=float, default=None, help="Manual vmax (only with --scale-mode manual)")

    args = ap.parse_args()

    base_dir = args.dir if args.dir is not None else default_base_dir(args.root)
    if not os.path.isdir(base_dir):
        raise SystemExit(f"Base directory not found: {base_dir}")

    z_slices = _parse_csv_floats(args.z_slices)

    run(
        dim_choice=args.dim,
        base_dir=base_dir,
        Nx=args.Nx,
        save=args.save,
        outdir=args.out,
        do_residual=args.residual,
        blur_sigma=args.blur_sigma,
        z_slices=z_slices,
        cmap_name=args.cmap,
        max_spy_n=args.max_spy_n,
        scale_mode=args.scale_mode,
        vmin_arg=args.vmin,
        vmax_arg=args.vmax,
    )


if __name__ == "__main__":
    main()
