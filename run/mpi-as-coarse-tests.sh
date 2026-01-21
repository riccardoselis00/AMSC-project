#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # dd_project

# Executable (MPI version) - absolute path
EXE="${ROOT}/build/benches/bench_mpi_as_coarse"

# Output CSV - absolute path
OUT_DIR="${ROOT}/data/output/csv"
mkdir -p "${OUT_DIR}"
OUT="${OUT_DIR}/results_mpi_as_scaling.csv"

# Problem sizes
SIZES=(4000 8000 16000)

# Preconditioners
PRECS=("as" "as2")

# MPI process counts
PROCS=(1 8 16)

# Other parameters
OVERLAP=1
BLOCK_SIZE=32
TOL=1e-12
MAXIT=500000

# Choose launcher: default mpirun, allow override (e.g., LAUNCHER="srun -n")
LAUNCHER="${LAUNCHER:-mpirun -np}"

# Sanity checks
if [[ ! -x "${EXE}" ]]; then
  echo "ERROR: executable not found or not executable: ${EXE}" >&2
  echo "Build first: cmake --build ${ROOT}/build" >&2
  exit 1
fi

# Header (fixed)
echo "n,prec,nprocs,iters,time_setup,time_solve,total_time" > "${OUT}"

for N in "${SIZES[@]}"; do
  for PREC in "${PRECS[@]}"; do
    for P in "${PROCS[@]}"; do

      echo "Running n=${N} prec=${PREC} procs=${P} ..."

      LINE="$(
        ${LAUNCHER} "${P}" "${EXE}" \
          --n "${N}" \
          --prec "${PREC}" \
          --block-size "${BLOCK_SIZE}" \
          --overlap "${OVERLAP}" \
          --tol "${TOL}" \
          --maxit "${MAXIT}" \
        | tail -1
      )"

      # If your bench prints: n,prec,iters,time_setup,time_solve,total_time
      # then cut from field 3 onward (iters,...)
      METRICS="$(echo "${LINE}" | cut -d',' -f3-)"

      echo "${N},${PREC},${P},${METRICS}" >> "${OUT}"

    done
  done
done

echo "Done. Results written to ${OUT}"
