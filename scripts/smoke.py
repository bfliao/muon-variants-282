#!/usr/bin/env python
"""Smoke runs — which ARE the timing runs (Gate-0 answer 13).

For each optimizer of one study (all 5, plus a ManifoldMuon cap-10 variant
while the cap is undecided), run 200 steps on the REAL data through
train.run (phase "smoke", eval every 50 steps, the real configs' literature
default lr), then assert:

* the result JSON exists with status "completed",
* the validation loss fell from the first to the last eval record,
* the param-map fractions match the expected split for (optimizer, arch).

Timing: wall-clock from inside the training loop is not accessible from
here, so the simplest correct approach is used — train.run is timed for a
200-step config AND separately for a 20-step config of the same world, and
per-step time = (t200 - t20) / 180. The subtraction removes the shared
one-off costs (model build, data open, final eval); the 200-step run's four
mid-run evals remain inside the numerator, which slightly OVERSTATES s/step
relative to real runs (they eval every 100/176 steps) — a conservative
planning number. Steps 1-20 (warm-up effects) are excluded by construction.

Each timed pair appends one row to hardware-results.csv:
hostname, device, study, optimizer, cap, s_per_step, timestamp.

Usage:
  uv run python scripts/smoke.py --study mlp128
  uv run python scripts/smoke.py --study dense --optimizer manifold
  uv run python scripts/smoke.py --study moe --manifold-cap10   # cap-10 only

NOTE: the smoke configs' hashes are stable, so a re-invocation finds the
results done and train.run returns instantly — the timing would be garbage.
Already-done smoke runs are therefore re-ASSERTED but not re-timed; delete
results/<study>/<hash>.json (printed below) to re-time on this machine.
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from muonarch import runner, studies, train  # noqa: E402

SMOKE_STEPS = 200
BASELINE_STEPS = 20
EVAL_EVERY = 50
CSV_FIELDS = ("hostname", "device", "study", "optimizer", "cap",
              "s_per_step", "timestamp")


def smoke_config(study: str, opt: str, steps: int, dual_cap: int | None = None) -> dict:
    """A real-data smoke config: literature defaults, phase 'smoke',
    ``steps`` total steps, eval every 50. The muon/manifold auxiliary AdamW
    group uses AdamW's literature defaults (tuning has not happened yet)."""
    sv = studies.load_study(study)
    group = None
    if opt in studies.AUX_ADAMW_OPTIMIZERS:
        group = (float(sv["optimizers"]["adamw"]["lr"]),
                 float(sv["optimizers"]["adamw"]["wd"]))
    ev = dict(sv["eval"], every_steps=EVAL_EVERY)
    cfg = studies.run_config(
        sv, opt, "smoke", studies.cd_defaults(sv, opt), seed=0,
        adamw_group=group, budget_base_steps=steps, eval=ev)
    if dual_cap is not None:
        cfg["manifold"] = dict(cfg["manifold"], dual_cap=dual_cap)
    return cfg


def fraction_ok(opt: str, arch: str, muon_frac: float) -> bool:
    """Expected Muon-updated parameter share (PLAN Protocol: >99% MLP, ~69%
    dense once the input layer is in the Muon group; MuonAll touches
    everything, AdamW nothing)."""
    if opt == "adamw":
        return muon_frac == 0.0
    if opt == "muonall":
        return muon_frac == 1.0
    if arch == "mlp":
        return muon_frac > 0.99
    if arch == "dense":
        return 0.60 < muon_frac < 0.80
    return 0.60 < muon_frac < 0.95   # moe: more expert params, still tied emb


def check_run(root: Path, cfg: dict) -> None:
    """Post-run assertions on the 200-step smoke run."""
    h = runner.config_hash(cfg)
    res_path = runner.result_path(root, cfg)
    assert res_path.is_file(), f"no result for smoke run {h}"
    res = json.loads(res_path.read_text())
    assert res["status"] == "completed", \
        f"smoke run {h} ended {res['status']!r} (diverged at default lr?)"
    evals = []
    for line in runner.log_path(root, cfg).read_text().splitlines():
        rec = json.loads(line)
        if rec.get("t") == "eval":
            evals.append(rec)
        if rec.get("t") == "start":
            frac = rec["fractions"]["muon"]
            assert fraction_ok(cfg["optimizer"], cfg["arch"], frac), (
                f"{cfg['optimizer']}/{cfg['arch']}: muon fraction {frac:.4f} "
                f"outside the expected range")
    assert len(evals) >= 2, f"smoke run {h}: fewer than 2 eval records"
    first, last = evals[0]["val_loss"], evals[-1]["val_loss"]
    assert last < first, (
        f"smoke run {h}: val loss did not fall ({first:.4f} -> {last:.4f})")
    print(f"  OK {h}: val {first:.4f} -> {last:.4f}, "
          f"muon fraction checks passed")


def append_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--study", required=True, choices=studies.STUDIES)
    ap.add_argument("--optimizer", choices=studies.OPTIMIZERS,
                    help="run only this optimizer (manifold: both caps)")
    ap.add_argument("--manifold-cap10", action="store_true",
                    help="run ONLY the ManifoldMuon cap-10 variant")
    ap.add_argument("--device", default=None)
    ap.add_argument("--root", default=Path(__file__).resolve().parent.parent,
                    type=Path)
    ap.add_argument("--csv", default=None, type=Path,
                    help="default: <root>/hardware-results.csv")
    args = ap.parse_args(argv)

    import torch

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    csv_path = args.csv or (args.root / "hardware-results.csv")
    sv = studies.load_study(args.study)

    # (optimizer, cap) units: every optimizer at its default settings,
    # plus Manifold at cap 10 while the cap is undecided (Gate-0 answer 13).
    default_cap = int(sv["optimizers"]["manifold"].get("dual_cap", 5))
    units: list[tuple[str, int | None]] = [
        (o, default_cap if o == "manifold" else None) for o in studies.OPTIMIZERS]
    units.append(("manifold", 10))
    if args.manifold_cap10:
        units = [("manifold", 10)]
    elif args.optimizer:
        units = [u for u in units if u[0] == args.optimizer]

    failures: list[str] = []
    for opt, cap in units:
        label = f"{opt}" + (f" (cap {cap})" if cap is not None else "")
        print(f"[smoke] {args.study} / {label} on {device}")
        try:
            cfg20 = smoke_config(args.study, opt, BASELINE_STEPS, dual_cap=cap)
            cfg200 = smoke_config(args.study, opt, SMOKE_STEPS, dual_cap=cap)
            fresh = not (runner.is_done(args.root, cfg20)
                         or runner.is_done(args.root, cfg200))
            t0 = time.time()
            train.run(cfg20, args.root, device=device)
            t20 = time.time() - t0
            t0 = time.time()
            train.run(cfg200, args.root, device=device)
            t200 = time.time() - t0
            check_run(args.root, cfg200)
            if not fresh:
                print(f"  already done — NOT re-timed; delete "
                      f"{runner.result_path(args.root, cfg200)} and "
                      f"{runner.result_path(args.root, cfg20)} to re-time")
                continue
            s_per_step = (t200 - t20) / (SMOKE_STEPS - BASELINE_STEPS)
            print(f"  t20={t20:.1f}s t200={t200:.1f}s -> {s_per_step:.4f} s/step")
            append_csv(csv_path, {
                "hostname": socket.gethostname(),
                "device": device,
                "study": args.study,
                "optimizer": opt,
                "cap": "" if cap is None else cap,
                "s_per_step": f"{s_per_step:.6f}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001 — a failed unit must not stop the rest
            failures.append(f"{args.study}/{label}: {e}")
            print(f"  FAILED {args.study}/{label}: {e}", file=sys.stderr)
    print(f"[smoke] done; timing rows in {csv_path}")
    if failures:
        print(f"[smoke] {len(failures)} unit(s) FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
