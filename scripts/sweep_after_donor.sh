#!/bin/bash
# Run a Phase-II-donor-dependent study: run what's ready now (CD track),
# wait for the donor's runner (pid $2) to exit, then run again so the
# inherited Phase-II track and the shared tail complete.
set -o pipefail
cd "$(dirname "$0")/.."
s="$1"; donor_pid="$2"
run() { uv run python -m muonarch.runner --study "$s" 2>&1 | grep -vE "UserWarning|svdvals" | tee -a "logs/sweep_$s.log"; }
echo "=== $s start $(date) ==="
run
while kill -0 "$donor_pid" 2>/dev/null; do sleep 60; done
echo "=== $s donor finished $(date) ==="
run
run   # one more pass in case the tail became ready late
uv run python -m muonarch.runner --study "$s" --status 2>&1 | tail -25
echo "=== $s end $(date) ==="
