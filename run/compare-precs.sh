#!/usr/bin/env bash
set -euo pipefail

EXE=./build/benches/bench_pcg_sequential  
OUT=results_seq_precs.csv        

SIZES=(10000 20000 40000 80000 160000)

PRECS=("identity" "diag_jacobi" "blockjac" "as")

BLOCK_SIZE=16
OVERLAP=1
TOL=1e-12
MAXIT=500000

echo "n,prec,iters,residual,time_setup,time_solve, total time" > "$OUT"

for N in "${SIZES[@]}"; do
  for PREC in "${PRECS[@]}"; do

    echo "Running N=$N prec=$PREC ..."

    LINE=$(
      $EXE \
        --n $N \
        --prec $PREC \
        --tol $TOL \
        --maxit $MAXIT \
      | tail -1   
    )
    echo "$LINE" >> "$OUT"
  done
done

echo "Done. Results saved in $OUT"