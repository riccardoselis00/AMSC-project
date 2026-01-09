#!/usr/bin/env bash
set -euo pipefail

# Executable (MPI version)
EXE=./build/benches/bench_mpi_as_coarse

# Output CSV
OUT=data/output/csv/results_mpi_as_overlap_sweep_n160k.csv
mkdir -p "$(dirname "$OUT")"

# Fixed problem size
N=80000

# Preconditioners
PRECS=("as" "as2")

# MPI process counts
PROCS=(1 2 4 8 16)

# Sweep overlaps (reasonable small-range)

# - If you want more, extend to 6 or 8.

OVERLAPS=(1 2 4 8 16 32 64)

# Other parameters (unchanged)
BLOCK_SIZE=32
TOL=1e-12
MAXIT=500000

# Header (match what we write)
echo "n,prec,nprocs,overlap,block_size,iters,time_setup,time_solve,total_time" > "$OUT"

for OV in "${OVERLAPS[@]}"; do
  for PREC in "${PRECS[@]}"; do
    for P in "${PROCS[@]}"; do

      echo "Running n=$N prec=$PREC procs=$P overlap=$OV block_size=$BLOCK_SIZE ..."

      LINE=$(
        mpirun -np "$P" "$EXE" \
          --n "$N" \
          --prec "$PREC" \
          --block-size "$BLOCK_SIZE" \
          --overlap "$OV" \
          --tol "$TOL" \
          --maxit "$MAXIT" \
        | tail -1
      )

      # Expect LINE like: n,prec,nprocs,iters,time_setup,time_solve,total_time
      # Strip the first 3 fields (n,prec,nprocs) since we already provide them:
      METRICS=$(echo "$LINE" | cut -d',' -f4-)

      echo "$N,$PREC,$P,$OV,$BLOCK_SIZE,$METRICS" >> "$OUT"

    done
  done
done

echo "Done. Results written to $OUT"
