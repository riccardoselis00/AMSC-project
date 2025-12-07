#!/usr/bin/env bash
set -euo pipefail

# Executable
EXE=./build/benches/bench_mpi_as_coarse

# Output CSV
OUT=data/output/csv/strong_scaling.csv

# Fixed problem sizes for strong scaling
SIZES=(40000 80000 160000 320000)

# Number of MPI processes to test
PROCS=(1 2 4 8 16)

# Preconditioners to test
PRECS=("as2")

# Solver params
TOL=1e-16
MAXIT=500000

echo "n,prec,nprocs,iters,residual,time_setup,time_solve,total_time" > "$OUT"

for N in "${SIZES[@]}"; do
    for PREC in "${PRECS[@]}"; do
        for P in "${PROCS[@]}"; do

            echo "Running STRONG scaling: n=$N prec=$PREC nprocs=$P"

            LINE=$(
                mpirun -np $P "$EXE" \
                    --n "$N" \
                    --prec "$PREC" \
                    --tol "$TOL" \
                    --maxit "$MAXIT" \
                | tail -1
            )

            echo "$LINE" >> "$OUT"
        done
    done
done

echo "Done. Strong scaling results saved in $OUT"
