#!/usr/bin/env bash
set -euo pipefail

# Executable (MPI version)
EXE=./build/benches/bench_mpi_as_coarse

# Output CSV
OUT=data/output/csv/results_mpi_as_scaling.csv

# Problem sizes
SIZES=(40000 80000 160000 320000)

# Preconditioners: baseline, MPI, coarse, and MPI + coarse
PRECS=("as" "as2")

# MPI process counts

#PROCS=(1)
PROCS=(1 2 4 8 16)

# Other parameters
OVERLAP=1
BLOCK_SIZE=32
TOL=1e-12
MAXIT=500000

# Header
echo "n,prec,nprocs,nprocs,iters,time_setup,time_solve,total_time" > "$OUT"

for N in "${SIZES[@]}"; do
  for PREC in "${PRECS[@]}"; do
    for P in "${PROCS[@]}"; do

      echo "Running n=$N prec=$PREC procs=$P ..."

      # Use mpirun or srun depending on your environment
      LINE=$(
        mpirun -np "$P" "$EXE" \
          --n "$N" \
          --prec "$PREC" \
          --block-size "$BLOCK_SIZE" \
          --overlap "$OVERLAP" \
          --tol "$TOL" \
          --maxit "$MAXIT" \
        | tail -1
      )

      # CSV row format remains consistent with sequential tests
      METRICS=$(echo "$LINE" | cut -d',' -f3-)

      echo "$N,$PREC,$P,$METRICS" >> "$OUT"

    done
  done
done

echo "Done. Results written to $OUT"
