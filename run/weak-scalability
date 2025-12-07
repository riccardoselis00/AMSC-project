#!/usr/bin/env bash
set -euo pipefail

# Executable
EXE=./build/benches/bench_mpi_as_coarse

# Output CSV
OUT=data/output/csv/weak_scaling.csv

# Baseline problem size per process
BASE_N=40000

# MPI process counts
PROCS=(1 2 4 8 16)

# Preconditioners
PRECS=("as2")

# Solver params
TOL=1e-16
MAXIT=500000

echo "n,prec,nprocs,iters,residual,time_setup,time_solve,total_time" > "$OUT"

for PREC in "${PRECS[@]}"; do
    for P in "${PROCS[@]}"; do

        # Scale N proportionally to P
        N=$((BASE_N * P))

        echo "Running WEAK scaling: n=$N prec=$PREC nprocs=$P"

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

echo "Done. Weak scaling results saved in $OUT"
