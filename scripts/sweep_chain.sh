#!/bin/bash
# One runner process at a time (CIFAR-5m train is ~5.3 GB per process; three
# MLP processes in parallel exhausted 36 GB RAM on Nov 17). Each study runs
# to completion before the next starts; donors finish first by construction.
set -o pipefail
cd "$(dirname "$0")/.."
for s in "$@"; do
  echo "=== $s start $(date) ==="
  uv run python -m muonarch.runner --study "$s" 2>&1 | grep -vE "UserWarning|svdvals" | tee -a "logs/sweep_$s.log"
  uv run python -m muonarch.runner --study "$s" --status 2>&1 | tail -40
  echo "=== $s end $(date) ==="
done
