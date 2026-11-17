#!/bin/bash
# Phase 3: smoke + timing runs (200 steps, every optimizer x study, Manifold caps 5 and 10; cap 10 is included by default).
set -o pipefail
cd "$(dirname "$0")/.."
for s in mlp128 mlp256 mlp512 dense moe; do
  echo "=== $s $(date) ==="
  uv run python scripts/smoke.py --study "$s" 2>&1 | tee "logs/phase3_$s.log" || echo "STUDY $s FAILED"
done
uv run python scripts/timing.py && echo "PHASE3 DONE $(date)"
