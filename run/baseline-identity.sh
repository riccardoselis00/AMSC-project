#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# run_baseline_sweep.sh
#
# Runs baseline PCG + Identity for multiple dimensions/sizes.
# Writes a single CSV file with (dim,nx,ny,nz,unknowns,nnz,iters,times...).
#
# Intended to be launched from repo root OR from build/.
# ============================================================

# -------- locate repo root robustly (script directory) --------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# -------- executable path (adjust if your target name differs) --------
# Example: build/benches/bench_baseline_pcg_identity
EXE_DEFAULT="${REPO_ROOT}/build/benches/bench_baseline_identity_solver"
EXE="${EXE:-$EXE_DEFAULT}"

# -------- output CSV --------
OUT_DEFAULT="${REPO_ROOT}/data/output/csv/baseline_identity_sweep.csv"
OUT="${OUT:-$OUT_DEFAULT}"
mkdir -p "$(dirname "$OUT")"

# -------- solver / PDE params --------
TOL="${TOL:-1e-12}"
MAXIT="${MAXIT:-500000}"
REPEAT="${REPEAT:-1}"
MU="${MU:-1.0}"
C="${C:-0.0}"

# -------- sweep definition --------
# If you want per-dimension different Nx ranges, use separate arrays below.
DIMS=(${DIMS:-1 2 3})

# "Nx list" used when you want isotropic grids: n=(Nx) or (Nx,Nx) or (Nx,Nx,Nx)
# Choose sane values for 3D (unknowns grow as Nx^3).
SIZES_1D=(${SIZES_1D:-2000 5000 10000})
SIZES_2D=(${SIZES_2D:-50 100 150 200})
SIZES_3D=(${SIZES_3D:-10 15 20 25})

# Optional: explicit non-isotropic cases (uncomment and edit).
# Format: "dim:nx,ny,nz"
# These will run IN ADDITION to the isotropic sweeps above.
EXTRA_CASES=(
  # "2:400,200"
  # "3:40,30,20"
)

# -------- info --------
echo "== Baseline sweep =="
echo "EXE    : $EXE"
echo "OUT    : $OUT"
echo "TOL    : $TOL"
echo "MAXIT  : $MAXIT"
echo "REPEAT : $REPEAT"
echo "MU,C   : $MU, $C"
echo

if [[ ! -x "$EXE" ]]; then
  echo "ERROR: executable not found or not executable: $EXE" >&2
  echo "Build it first, or set EXE=/path/to/executable" >&2
  exit 1
fi

# Start fresh CSV (overwrite) then append
rm -f "$OUT"

run_case() {
  local dim="$1"
  shift
  echo "Running: dim=$dim $*"
  "$EXE" \
    --dim="$dim" \
    --tol="$TOL" \
    --maxit="$MAXIT" \
    --repeat="$REPEAT" \
    --mu="$MU" \
    --c="$C" \
    --csv="$OUT" \
    --append \
    "$@"
}

# -------- isotropic sweeps --------
for dim in "${DIMS[@]}"; do
  case "$dim" in
    1)
      for Nx in "${SIZES_1D[@]}"; do
        run_case 1 --Nx="$Nx"
      done
      ;;
    2)
      for Nx in "${SIZES_2D[@]}"; do
        run_case 2 --Nx="$Nx"
      done
      ;;
    3)
      for Nx in "${SIZES_3D[@]}"; do
        run_case 3 --Nx="$Nx"
      done
      ;;
    *)
      echo "Skipping unsupported dim=$dim" >&2
      ;;
  esac
done

# -------- extra explicit n-cases --------
if (( ${#EXTRA_CASES[@]} > 0 )); then
  echo
  echo "== Running EXTRA_CASES =="
  for spec in "${EXTRA_CASES[@]}"; do
    # split at first ':'
    dim="${spec%%:*}"
    n="${spec#*:}"
    run_case "$dim" --n="$n"
  done
fi

echo
echo "Done."
echo "CSV: $OUT"
