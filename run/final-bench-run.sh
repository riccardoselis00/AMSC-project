#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# run_final_physical_identity_vs_as2.sh
#
# Baseline:     PCG + Identity ONLY with MPI np=1
# Comparison:   PCG_MPI + AS2 with np in {1,2,4,8,16}
#
# Runs for dim=1,2,3 over size lists.
# Writes ONE CSV (appended by the executable itself).
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---- MPI physical benchmark executable (identity/as2) ----
# Example: build/benches/bench_mpi_physical_as_coarse
EXE_DEFAULT="${REPO_ROOT}/build/benches/bench_final_identity_vs_AS2"
EXE="${EXE:-$EXE_DEFAULT}"

# ---- Output CSV ----
OUT_DEFAULT="${REPO_ROOT}/data/output/csv/final_physical_identity_vs_as2.csv"
OUT="${OUT:-$OUT_DEFAULT}"
mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"

# ---- Solver / PDE params ----
TOL="${TOL:-1e-12}"
MAXIT="${MAXIT:-500000}"
REPEAT="${REPEAT:-1}"
MU="${MU:-1.0}"
C="${C:-0.0}"

# ---- AS2 params ----
NPARTS="${NPARTS:-32}"
OVERLAP="${OVERLAP:-1}"

# ---- Sweep definition ----
# Identity baseline ONLY at np=1
BASELINE_NP=1

# AS2 sweep
PROCS_AS2=(1 2 4 8 16)

# Per-dimension sizes
# 1D uses --n="Nx"
SIZES_1D=(40000 80000 160000)

# 2D uses --n="Nx,Ny"
SIZES_2D=("400,400" "800,800" "1600,1600")

# 3D uses --n="Nx,Ny,Nz"
SIZES_3D=("40,40,40" "80,80,80" "160,160,160")

echo "== Final physical benchmark: Identity@np=1 vs AS2@np={1..16} =="
echo "EXE      : $EXE"
echo "OUT      : $OUT"
echo "TOL      : $TOL"
echo "MAXIT    : $MAXIT"
echo "REPEAT   : $REPEAT"
echo "MU,C     : $MU, $C"
echo "NPARTS   : $NPARTS"
echo "OVERLAP  : $OVERLAP"
echo

if [[ ! -x "$EXE" ]]; then
  echo "ERROR: executable not found or not executable: $EXE" >&2
  exit 1
fi

run_one() {
  local np="$1"
  local dim="$2"
  local prec="$3"
  local nstr="$4"

  echo "Running: np=$np dim=$dim n=$nstr prec=$prec"

  mpirun -np "$np" "$EXE" \
    --dim "$dim" \
    --n "$nstr" \
    --prec "$prec" \
    --nparts "$NPARTS" \
    --overlap "$OVERLAP" \
    --tol "$TOL" \
    --maxit "$MAXIT" \
    --repeat "$REPEAT" \
    --mu "$MU" \
    --c "$C" \
    --csv "$OUT" \
    --append \
  | tail -1
}

run_dim_set() {
  local dim="$1"; shift
  local -a ns=( "$@" )

  for nstr in "${ns[@]}"; do
    # ---- baseline identity ONLY at np=1 ----
    run_one "$BASELINE_NP" "$dim" "identity" "$nstr"

    # ---- AS2 sweep ----
    for np in "${PROCS_AS2[@]}"; do
      run_one "$np" "$dim" "as2" "$nstr"
    done
  done
}

# 1D / 2D / 3D sweeps
run_dim_set 1 "${SIZES_1D[@]}"
run_dim_set 2 "${SIZES_2D[@]}"
run_dim_set 3 "${SIZES_3D[@]}"

echo
echo "Done."
echo "CSV: $OUT"
