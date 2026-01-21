#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # dd_project

# Executable (absolute path)
EXE="${ROOT}/build/benches/bench_mpi_as_coarse"

# Output CSV (absolute path)
OUT_DIR="${ROOT}/data/output/csv"
mkdir -p "${OUT_DIR}"
OUT="${OUT_DIR}/strong_scaling.csv"

# Fixed problem sizes for strong scaling
SIZES=(4000 8000 16000 32000)

# Number of MPI processes to test
PROCS=(1 8 16)

# Preconditioners to test
PRECS=("as2")

# Solver params
TOL=1e-16
MAXIT=500000

# Launcher (override if needed)
# Examples:
#   LAUNCHER="mpirun -np"   bash run/strong-scalability.sh
#   LAUNCHER="srun -n"      bash run/strong-scalability.sh
LAUNCHER="${LAUNCHER:-mpirun -np}"

# Sanity check
if [[ ! -f "${EXE}" ]]; then
  echo "ERROR: executable not found: ${EXE}" >&2
  echo "Build it: cmake --build ${ROOT}/build" >&2
  exit 1
fi
if [[ ! -x "${EXE}" ]]; then
  echo "ERROR: executable is not executable: ${EXE}" >&2
  echo "Fix with: chmod +x ${EXE}" >&2
  exit 1
fi

echo "n,prec,nprocs,iters,residual,time_setup,time_solve,total_time" > "${OUT}"

for N in "${SIZES[@]}"; do
  for PREC in "${PRECS[@]}"; do
    for P in "${PROCS[@]}"; do

      echo "Running STRONG scaling: n=${N} prec=${PREC} nprocs=${P}"

      LINE="$(
        ${LAUNCHER} "${P}" "${EXE}" \
          --n "${N}" \
          --prec "${PREC}" \
          --tol "${TOL}" \
          --maxit "${MAXIT}" \
        | tail -1
      )"

      echo "${LINE}" >> "${OUT}"
    done
  done
done

echo "Done. Strong scaling results saved in ${OUT}"
