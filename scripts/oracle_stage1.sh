#!/bin/zsh
# Phase 1, Stage 1 of the oracle check: seed 42, LR sweeps.
#   AdamW lr in {3e-4, 1e-3, 3e-3, 1e-2}, wd 0
#   Manifold Muon lr in {0.05, 0.1, 0.2}
# 3 epochs, batch 1024 (vendored defaults), 100 dual steps (vendored default).
# CWD = logs/oracle so ./data and ./results land there. Skips completed runs
# (the vendored script writes a deterministic pickle name on completion).
set -e -o pipefail
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

for lr in 0.0003 0.001 0.003 0.01; do run adam "$lr" 0.0 42; done
for lr in 0.05 0.1 0.2; do run manifold_muon "$lr" 0.0 42; done
echo "STAGE 1 COMPLETE"
