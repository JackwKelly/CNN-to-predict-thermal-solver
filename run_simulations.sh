#!/usr/bin/env bash
# Run simulate.py for ids 0..N_SIMS-1, N_PARALLEL processes at a time.
#
# Usage:  ./run_simulations.sh [N_SIMS=2000] [N_PARALLEL=20]
# Env:    PYTHON=python3   interpreter to use
#         OUT_DIR=data/csv output directory
#
# NOTE: existing sim_*.csv files in OUT_DIR are deleted first so stale runs
# can never be mixed into the dataset.
set -euo pipefail

N_SIMS="${1:-2000}"  #takes the 1st argument or 2000 if not given
N_PARALLEL="${2:-20}" # takes the 2nd argument or 20 if not given
PYTHON="${PYTHON:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-$HERE/data/csv}"

# One thread per process, otherwise 20 numpy processes oversubscribe the CPU.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1



if ! [[ "$N_SIMS" =~ ^[0-9]+$ ]]; then
    echo "error: N_SIMS must be a positive integer, got '$N_SIMS'" >&2
    exit 1
fi

if (( N_SIMS < 3 )); then
    echo "warning: data partitioner will fail with less than 3 runs, Setting number of runs to 3" >&2
    N_SIMS=3
fi

mkdir -p "$OUT_DIR"  # making the output directory
rm -f "$OUT_DIR"/sim_*.csv # removes old data

echo "Running $N_SIMS simulations, $N_PARALLEL in parallel -> $OUT_DIR"
start=$(date +%s)

# xargs exits non-zero if any simulation fails, which stops the script (set -e).
seq 0 $((N_SIMS - 1)) | xargs -P "$N_PARALLEL" -I{} \
    "$PYTHON" "$HERE/simulate.py" --id {} --out-dir "$OUT_DIR"

echo "Done: $(ls "$OUT_DIR"/sim_*_final.csv | wc -l) simulations in $(( $(date +%s) - start ))s"
echo "Next: python concat_to_hdf5.py"
