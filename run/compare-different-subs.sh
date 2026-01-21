#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # dd_project
OUT_CSV="${ROOT}/data/output/csv"
mkdir -p "${OUT_CSV}"

# Output CSV
OUT="${OUT_CSV}/results_seq_as_blocks.csv"

# Executable (absolute path, robust)
EXE="${ROOT}/build/benches/bench_pcg_sequential"

# Sanity checks
if [[ ! -x "${EXE}" ]]; then
  echo "ERROR: executable not found or not executable: ${EXE}" >&2
  echo "Did you build the project? (cmake --build build)" >&2
  exit 1
fi

# Problem sizes
SIZES=(1000 2000 4000 8000 16000)

# Block sizes for Additive Schwarz
BLOCKS=(4 8 16 32)

# Other solver parameters
OVERLAP=1
TOL=1e-16
MAXIT=500000

# ----------------------------------------------------------
# CSV header
# ----------------------------------------------------------
echo "n,prec,block_size,iters,residual,time_setup,time_solve,total_time" > "${OUT}"

# ----------------------------------------------------------
# For each N, run:
#   1) identity baseline  (block_size = 0)
#   2) additive schwarz for block sizes ∈ BLOCKS
# ----------------------------------------------------------
for N in "${SIZES[@]}"; do

  echo "Running N=${N} prec=identity ..."

  LINE="$(
    "${EXE}" \
      --n "${N}" \
      --prec identity \
      --tol "${TOL}" \
      --maxit "${MAXIT}" \
    | tail -1
  )"

  METRICS="$(echo "${LINE}" | cut -d',' -f3-)"
  echo "${N},identity,0,${METRICS}" >> "${OUT}"

  for BLOCK in "${BLOCKS[@]}"; do
    echo "Running N=${N} prec=as block_size=${BLOCK} ..."

    LINE="$(
      "${EXE}" \
        --n "${N}" \
        --prec as \
        --block-size "${BLOCK}" \
        --overlap "${OVERLAP}" \
        --tol "${TOL}" \
        --maxit "${MAXIT}" \
      | tail -1
    )"

    METRICS="$(echo "${LINE}" | cut -d',' -f3-)"
    echo "${N},as,${BLOCK},${METRICS}" >> "${OUT}"
  done

done

echo "Done. Results saved in ${OUT}"
