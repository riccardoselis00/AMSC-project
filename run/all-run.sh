#!/usr/bin/env bash
set -Eeuo pipefail

# ------------------------------------------------------------------------------
# Run all benchmark scripts in this folder in a fixed order.
# Logs stdout/stderr of each script to run/logs/<timestamp>_<script>.log
#
# Usage:
#   ./all.sh                 # stop on first failure
#   ./all.sh --continue      # run all even if some fail
#   ./all.sh --parallel 3    # run up to 3 scripts at once (careful with MPI)
# ------------------------------------------------------------------------------

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="${HERE}/logs"
mkdir -p "${LOGDIR}"

CONTINUE_ON_FAIL=0
PARALLEL=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --continue) CONTINUE_ON_FAIL=1; shift ;;
    --parallel) PARALLEL="${2:-1}"; shift 2 ;;
    -h|--help)
      sed -n '1,120p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

timestamp() { date +"%Y%m%d_%H%M%S"; }

# Fixed order (edit if you want a different sequence)
SCRIPTS=(
  "baseline-identity.sh"
  "compare-different-subs.sh"
  #"compare-precs.sh"
  "mpi-as-coarse-tests.sh"
  # "mpi-as-coarse-tests-blocks.sh"
  # "mpi-as-coarse-tests-overlap.sh"
  "strong-scalability.sh"
  # "weak-scalability.sh"
  "final-bench-run.sh"
)

# Ensure scripts exist + executable
missing=0
for s in "${SCRIPTS[@]}"; do
  if [[ ! -f "${HERE}/${s}" ]]; then
    echo "Missing: ${HERE}/${s}" >&2
    missing=1
  fi
done
[[ $missing -eq 1 ]] && exit 1

for s in "${SCRIPTS[@]}"; do
  if [[ ! -x "${HERE}/${s}" ]]; then
    chmod +x "${HERE}/${s}" || true
  fi
done

run_one() {
  local s="$1"
  local ts
  ts="$(timestamp)"
  local log="${LOGDIR}/${ts}_${s%.sh}.log"

  echo "==> Running: ${s}"
  echo "    Log: ${log}"

  # Run inside the run/ folder so relative paths in scripts keep working
  (
    cd "${HERE}"
    bash "./${s}"
  ) >"${log}" 2>&1
}

FAILS=0

if [[ "${PARALLEL}" -le 1 ]]; then
  # Sequential (safest for MPI jobs / shared outputs)
  for s in "${SCRIPTS[@]}"; do
    if run_one "${s}"; then
      echo "==> OK: ${s}"
    else
      echo "==> FAIL: ${s} (see logs/)" >&2
      FAILS=$((FAILS + 1))
      if [[ "${CONTINUE_ON_FAIL}" -eq 0 ]]; then
        exit 1
      fi
    fi
  done
else
  # Parallel (use with caution: MPI scripts may contend for nodes/outputs)
  pids=()
  names=()

  wait_one() {
    local idx="$1"
    local pid="${pids[$idx]}"
    local name="${names[$idx]}"
    if wait "${pid}"; then
      echo "==> OK: ${name}"
    else
      echo "==> FAIL: ${name} (see logs/)" >&2
      FAILS=$((FAILS + 1))
      if [[ "${CONTINUE_ON_FAIL}" -eq 0 ]]; then
        exit 1
      fi
    fi
  }

  for s in "${SCRIPTS[@]}"; do
    # throttle
    while [[ "${#pids[@]}" -ge "${PARALLEL}" ]]; do
      wait_one 0
      pids=("${pids[@]:1}")
      names=("${names[@]:1}")
    done

    (
      run_one "${s}"
    ) &
    pids+=("$!")
    names+=("${s}")
  done

  # wait remaining
  for i in "${!pids[@]}"; do
    wait_one "${i}"
  done
fi

echo "============================================================"
echo "Done. Total failed scripts: ${FAILS}"
echo "Logs in: ${LOGDIR}"
echo "============================================================"

exit $([[ "${FAILS}" -eq 0 ]] && echo 0 || echo 1)
