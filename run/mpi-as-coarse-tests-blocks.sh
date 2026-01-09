#!/usr/bin/env bash
set -euo pipefail

# Executable (MPI version)
EXE=./build/benches/bench_mpi_as_coarse

# Output CSV
OUT=data/output/csv/results_mpi_as_scaling_blocks.csv
mkdir -p "$(dirname "$OUT")"

# Problem sizes
SIZES=(40000 80000 160000 320000)

# Preconditioners
PRECS=("as" "as2")

# MPI process counts
PROCS=(1 2 4 8 16)

# Sweep block sizes (nparts per rank)
# Pick values that divide n_loc nicely for most cases (not required, but cleaner).
BLOCK_SIZES=(1 2 4 8 16 32)

# Other parameters
OVERLAP=1
TOL=1e-12
MAXIT=500000

# Header
echo "n,prec,nprocs,block_size,overlap,iters,time_setup,time_solve,total_time" > "$OUT"

for N in "${SIZES[@]}"; do
  for PREC in "${PRECS[@]}"; do
    for P in "${PROCS[@]}"; do
      for BS in "${BLOCK_SIZES[@]}"; do

        echo "Running n=$N prec=$PREC procs=$P block_size=$BS overlap=$OVERLAP ..."

        LINE=$(
          mpirun -np "$P" "$EXE" \
            --n "$N" \
            --prec "$PREC" \
            --block-size "$BS" \
            --overlap "$OVERLAP" \
            --tol "$TOL" \
            --maxit "$MAXIT" \
          | tail -1
        )

        # Your program prints:
        # n_global,prec,nprocs,iters,time_setup,time_solve,total_time
        # We want: n,prec,nprocs,block_size,overlap,iters,time_setup,time_solve,total_time
        # So extract fields 4..7 = iters,time_setup,time_solve,total_time
        METRICS_ITS_TO_TOTAL=$(echo "$LINE" | cut -d',' -f4-7)

        echo "$N,$PREC,$P,$BS,$OVERLAP,$METRICS_ITS_TO_TOTAL" >> "$OUT"

      done
    done
  done
done

echo "Done. Results written to $OUT"
