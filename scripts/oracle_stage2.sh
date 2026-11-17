#!/bin/zsh
# Phase 1, Stage 2 of the oracle check: best LR of each optimizer on seeds 42-44.
# Usage: oracle_stage2.sh <adamw_best_lr> <manifold_best_lr>
set -e -o pipefail
ADAMW_LR=${1:?usage: oracle_stage2.sh <adamw_lr> <manifold_lr>}
MANIFOLD_LR=${2:?usage: oracle_stage2.sh <adamw_lr> <manifold_lr>}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/logs/oracle"
cd "$ROOT/logs/oracle"

run() {  # update lr wd seed
  local update=$1 lr=$2 wd=$3 seed=$4
  local pkl="results/update-${update}-lr-${lr}-wd-${wd}-seed-${seed}.pkl"
  if [[ -f $pkl ]]; then
    echo "SKIP (done): $pkl"
    return
  fi
  echo "RUN: update=$update lr=$lr wd=$wd seed=$seed"
  uv run --project "$ROOT" python "$ROOT/scripts/oracle_tml.py" \
    --update "$update" --lr "$lr" --wd "$wd" --epochs 3 --seed "$seed" \
    2>&1 | tee "run-${update}-lr-${lr}-wd-${wd}-seed-${seed}.log"
}

for seed in 42 43 44; do run adam "$ADAMW_LR" 0.0 "$seed"; done
for seed in 42 43 44; do run manifold_muon "$MANIFOLD_LR" 0.0 "$seed"; done
echo "STAGE 2 COMPLETE"
