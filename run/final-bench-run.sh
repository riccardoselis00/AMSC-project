#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# run_final_physical_identity_vs_as2.sh
#
# Baseline:     PCG + Identity ONLY with MPI np=1
# Comparison:   PCG_MPI + AS2 with np in {1,2,4,8,16}
#
# Now supports variable diffusion mu(x) to make 2D/3D harder
# without increasing Nx.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

EXE_DEFAULT="${REPO_ROOT}/build/benches/bench_final_identity_vs_AS2"
EXE="${EXE:-$EXE_DEFAULT}"

OUT_DEFAULT="${REPO_ROOT}/data/output/csv/final_physical_identity_vs_as2.csv"
OUT="${OUT:-$OUT_DEFAULT}"
mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"

# ---- Solver params ----
TOL="${TOL:-1e-12}"
MAXIT="${MAXIT:-500000}"
REPEAT="${REPEAT:-1}"

# ---- AS2 params ----
NPARTS="${NPARTS:-32}"
OVERLAP="${OVERLAP:-1}"

# ---- Baseline identity only at np=1 ----
BASELINE_NP=1

# ---- AS2 sweep ----
PROCS_AS2=(1 2 4 8 16)

# ---- sizes ----
SIZES_1D=(40000 80000 160000)
SIZES_2D=("100,100" "200,200" "400,00")     # keep 1600^2 only if you really can
SIZES_3D=("20,20,20" "40,40,40" "80,80,80") # 160^3 is huge; use only if ok

# ============================================================
# Difficulty knobs (IMPORTANT)
# We want: identity harder in 2D/3D so AS2 becomes effective.
# Best knob: strong diffusion contrast with mu(x).
# ============================================================

# Default forcing
F_TYPE="${F_TYPE:-const}"
F_AMP="${F_AMP:-1.0}"

# Dimension-specific diffusion regimes:
# 1D: already hard -> keep const
MU_TYPE_1D="${MU_TYPE_1D:-const}"
MU_1D="${MU_1D:-1.0}"

# 2D: make hard with layered diffusion
MU_TYPE_2D="${MU_TYPE_2D:-layer}"
MU_MIN_2D="${MU_MIN_2D:-1e-4}"
MU_MAX_2D="${MU_MAX_2D:-1.0}"
MU_POS_2D="${MU_POS_2D:-0.5}"

# 3D: milder contrast than 2D (3D cost)
MU_TYPE_3D="${MU_TYPE_3D:-inclusion}"
MU_MIN_3D="${MU_MIN_3D:-1e-3}"
MU_MAX_3D="${MU_MAX_3D:-1.0}"
MU_RADIUS_3D="${MU_RADIUS_3D:-0.25}"

# Checker parameters (if you use mu-type=checker)
MU_CELLS="${MU_CELLS:-8}"

# Reaction (keep 0, because big c usually makes it easier)
C_COEFF="${C_COEFF:-0.0}"

echo "== Final physical benchmark: Identity@np=1 vs AS2@np={1..16} =="
echo "EXE      : $EXE"
echo "OUT      : $OUT"
echo "TOL      : $TOL"
echo "MAXIT    : $MAXIT"
echo "REPEAT   : $REPEAT"
echo "NPARTS   : $NPARTS"
echo "OVERLAP  : $OVERLAP"
echo "F        : $F_TYPE (amp=$F_AMP)"
echo "C        : $C_COEFF"
echo "MU 1D    : type=$MU_TYPE_1D mu=$MU_1D"
echo "MU 2D    : type=$MU_TYPE_2D min=$MU_MIN_2D max=$MU_MAX_2D pos=$MU_POS_2D"
echo "MU 3D    : type=$MU_TYPE_3D min=$MU_MIN_3D max=$MU_MAX_3D radius=$MU_RADIUS_3D"
echo "MU cells : $MU_CELLS"
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
  shift 4

  echo "Running: np=$np dim=$dim n=$nstr prec=$prec $*"

  mpirun -np "$np" "$EXE" \
    --dim "$dim" \
    --n "$nstr" \
    --prec "$prec" \
    --nparts "$NPARTS" \
    --overlap "$OVERLAP" \
    --tol "$TOL" \
    --maxit "$MAXIT" \
    --repeat "$REPEAT" \
    --c "$C_COEFF" \
    --f-type "$F_TYPE" \
    --f-amp "$F_AMP" \
    "$@" \
    --csv "$OUT" \
    --append \
  | tail -1
}

run_dim_set() {
  local dim="$1"; shift
  local -a ns=( "$@" )

  for nstr in "${ns[@]}"; do
    # baseline identity only at np=1
    run_one "$BASELINE_NP" "$dim" "identity" "$nstr" "${DIM_EXTRA_ARGS[@]}"

    # AS2 sweep
    for np in "${PROCS_AS2[@]}"; do
      run_one "$np" "$dim" "as2" "$nstr" "${DIM_EXTRA_ARGS[@]}"
    done
  done
}

# # ---- Dim 1 ----
# DIM_EXTRA_ARGS=( --mu-type "$MU_TYPE_1D" --mu "$MU_1D" )
# run_dim_set 1 "${SIZES_1D[@]}"

# ---- Dim 2 ----
DIM_EXTRA_ARGS=(
  --mu-type "$MU_TYPE_2D"
  --mu-min "$MU_MIN_2D" --mu-max "$MU_MAX_2D" --mu-pos "$MU_POS_2D"
  --mu-cells "$MU_CELLS"
)
run_dim_set 2 "${SIZES_2D[@]}"

# ---- Dim 3 ----
DIM_EXTRA_ARGS=(
  --mu-type "$MU_TYPE_3D"
  --mu-min "$MU_MIN_3D" --mu-max "$MU_MAX_3D" --mu-radius "$MU_RADIUS_3D"
  --mu-cells "$MU_CELLS"
)
run_dim_set 3 "${SIZES_3D[@]}"

echo
echo "Done."
echo "CSV: $OUT"
