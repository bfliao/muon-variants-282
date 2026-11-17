#!/bin/bash
# Gate 3: MLP chain (128 -> 256 -> 512, donors first) or the dense study.
set -o pipefail
cd "$(dirname "$0")/.."
for s in "$@"; do
  echo "=== $s start $(date) ==="
  uv run python -m muonarch.runner --study "$s" 2>&1 | grep -vE "UserWarning|svdvals" | tee -a "logs/sweep_$s.log" || echo "STUDY $s FAILED"
  uv run python -m muonarch.runner --study "$s" --status 2>&1 | tail -15
  echo "=== $s end $(date) ==="
done
