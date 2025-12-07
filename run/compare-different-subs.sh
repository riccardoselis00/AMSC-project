#!/usr/bin/env bash
set -euo pipefail

# Executable
EXE=./build/benches/bench_pcg_sequential

# Output CSV
OUT=data/output/csv/results_seq_as_blocks.csv

# Problem sizes
SIZES=(10000 20000 40000 80000 160000)

# Block sizes for Additive Schwarz
BLOCKS=(4 8 16 32)

# Other solver parameters
OVERLAP=1
TOL=1e-16
MAXIT=500000

# ----------------------------------------------------------
# CSV header
# ----------------------------------------------------------
echo "n,prec,block_size,iters,residual,time_setup,time_solve,total_time" > "$OUT"


# ----------------------------------------------------------
# For each N, run:
#   1) identity baseline  (block_size = 0)
#   2) additive schwarz for block sizes ∈ BLOCKS
# ----------------------------------------------------------
for N in "${SIZES[@]}"; do

  echo "Running N=$N prec=identity ..."

  # Identity run (block_size = 0)
  LINE=$(
    "$EXE" \
      --n "$N" \
      --prec identity \
      --tol "$TOL" \
      --maxit "$MAXIT" \
    | tail -1
  )

  # Remove first two fields (n,prec)
  METRICS=$(echo "$LINE" | cut -d',' -f3-)

  # Add identity row with block_size = 0
  echo "$N,identity,0,$METRICS" >> "$OUT"



  # --------------------------
  # Now AS with varying blocks
  # --------------------------
  for BLOCK in "${BLOCKS[@]}"; do

    echo "Running N=$N prec=as block_size=$BLOCK ..."

    LINE=$(
      "$EXE" \
        --n "$N" \
        --prec as \
        --block-size "$BLOCK" \
        --overlap "$OVERLAP" \
        --tol "$TOL" \
        --maxit "$MAXIT" \
      | tail -1
    )

    METRICS=$(echo "$LINE" | cut -d',' -f3-)

    # Write CSV row: n,prec,block_size,metrics...
    echo "$N,as,$BLOCK,$METRICS" >> "$OUT"

  done

done

echo "Done. Results saved in $OUT"
