#!/usr/bin/env python
"""Compute-budget estimate from the measured smoke timings.

Reads hardware-results.csv (written by scripts/smoke.py — the smoke runs ARE
the timing runs, Gate-0 answer 13) and writes tables/compute_estimate.md:
hours per study and in total for the FULL plan, at Manifold dual-step cap 5
and cap 10, against the 30-hour window, with PLAN's rules applied (cap 10
only if it fits; cuts proposed in the plan's order).

Job enumeration: studies.planned_jobs(study) — the same enumeration
--dry-run totals use — under its documented scenario (no accepted cd change,
no edge trial, half-LR not promoted). Hours = sum over jobs of
executed steps x measured s/step for (study, optimizer):

* s/step comes from the newest hardware-results.csv row for that
  (study, optimizer) — restricted to one machine with --hostname if several
  laptops contributed rows.
* Manifold s/step at cap c: linear in the msign calls per step. With both
  cap-5 and cap-10 measurements, s(c) is the linear fit through them; with
  one measurement at cap c0, s(c) = s(c0) x (c + 1) / (c0 + 1) (at most
  cap ascent calls + the retraction per step — an upper bound, since early
  stopping only reduces real calls). The faithful rerun (cap 100) uses the
  same model, so its (large) cost is included and shown.
* A missing (study, optimizer) measurement falls back to the nearest
  measured study of the same architecture family, scaled by the ratio of
  analytic per-step FLOPs (runner.estimate_flops); such cells are marked ~
  in the table. If no measurement exists anywhere for an optimizer, its
  jobs are excluded and the table says so loudly.

Usage: uv run python scripts/timing.py [--csv F] [--out F] [--window-hours 30]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from muonarch import runner, studies  # noqa: E402

ARCH_FAMILY = {"mlp128": "mlp", "mlp256": "mlp", "mlp512": "mlp",
               "dense": "lm", "moe": "lm"}


def read_measurements(path: Path, hostname: str | None):
    """{(study, optimizer, cap_or_None): s_per_step} — newest row wins."""
    out: dict[tuple, float] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for row in csv.DictReader(f):
            if hostname and row["hostname"] != hostname:
                continue
            cap = int(row["cap"]) if row.get("cap") else None
            out[(row["study"], row["optimizer"], cap)] = float(row["s_per_step"])
    return out


def per_step_flops(study: str, opt: str) -> float:
    """Analytic per-step FLOPs of a default 1x run — the scaling key for the
    nearest-study fallback."""
    sv = studies.load_study(study)
    group = None
    if opt in ("muon", "manifold"):
        group = (float(sv["optimizers"]["adamw"]["lr"]),
                 float(sv["optimizers"]["adamw"]["wd"]))
    cfg = studies.run_config(sv, opt, "smoke", studies.cd_defaults(sv, opt),
                             adamw_group=group)
    return runner.estimate_flops(cfg) / sv["budget_base_steps"]


class Timings:
    def __init__(self, meas: dict):
        self.meas = meas
        self.approximated: set[tuple[str, str]] = set()
        self.missing: set[str] = set()

    def _base(self, study: str, opt: str, cap: int | None) -> float | None:
        """s/step for (study, opt) at a measured cap, with the nearest-study
        FLOP-scaled fallback."""
        v = self.meas.get((study, opt, cap))
        if v is not None:
            return v
        fam = ARCH_FAMILY[study]
        donors = [s for s in studies.STUDIES
                  if ARCH_FAMILY[s] == fam and (s, opt, cap) in self.meas]
        if not donors:
            return None
        donor = donors[0]
        self.approximated.add((study, opt))
        scale = per_step_flops(study, opt) / per_step_flops(donor, opt)
        return self.meas[(donor, opt, cap)] * scale

    def manifold_s(self, study: str, cap: int) -> float | None:
        s5 = self._base(study, "manifold", 5)
        s10 = self._base(study, "manifold", 10)
        if s5 is not None and s10 is not None:
            return s5 + (s10 - s5) * (cap - 5) / 5.0
        for c0, s0 in ((5, s5), (10, s10)):
            if s0 is not None:
                return s0 * (cap + 1) / (c0 + 1)
        return None

    def s_per_step(self, study: str, opt: str, cap_scenario: int,
                   job: dict) -> float | None:
        if opt != "manifold":
            return self._base(study, opt, None)
        cap = job["manifold"].get("dual_cap", 5)
        if job.get("phase") != "faithful":
            cap = cap_scenario
        return self.manifold_s(study, cap)


def study_hours(study: str, timings: Timings, cap_scenario: int,
                drop_manifold: bool = False):
    """(hours, manifold_hours) for the full planned job list."""
    total = man = 0.0
    for job in studies.planned_jobs(study):
        opt = job["optimizer"]
        if drop_manifold and opt == "manifold":
            continue
        s = timings.s_per_step(study, opt, cap_scenario, job)
        if s is None:
            timings.missing.add(f"{study}/{opt}")
            continue
        hours = runner._run_steps(job) * s / 3600.0
        total += hours
        if opt == "manifold":
            man += hours
    return total, man


def main(argv=None) -> int:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path, default=root / "hardware-results.csv")
    ap.add_argument("--out", type=Path, default=root / "tables" / "compute_estimate.md")
    ap.add_argument("--window-hours", type=float, default=30.0)
    ap.add_argument("--hostname", default=None,
                    help="use only this machine's rows (default: all rows, "
                         "newest per (study, optimizer, cap) wins)")
    args = ap.parse_args(argv)

    meas = read_measurements(args.csv, args.hostname)
    if not meas:
        print(f"ERROR: no timing rows in {args.csv} — run scripts/smoke.py "
              f"first (the smoke runs ARE the timing runs).", file=sys.stderr)
        return 1
    timings = Timings(meas)

    per_study: dict[int, dict[str, tuple[float, float]]] = defaultdict(dict)
    for cap in (5, 10):
        for study in studies.STUDIES:
            per_study[cap][study] = study_hours(study, timings, cap)
    totals = {cap: sum(h for h, _ in per_study[cap].values()) for cap in (5, 10)}

    lines = [
        "# Compute estimate — measured s/step x planned job list",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()} from "
        f"`{args.csv.name}`" + (f" (host {args.hostname})" if args.hostname else "")
        + f"; window {args.window_hours:.0f} h.",
        "",
        "Scenario: `studies.planned_jobs` (no accepted coordinate-descent "
        "change, no edge trials, half-LR not promoted) — "
        + ", ".join(f"{s}: {len(studies.planned_jobs(s))}" for s in studies.STUDIES)
        + f" = {sum(len(studies.planned_jobs(s)) for s in studies.STUDIES)} runs. "
        "Manifold msign cost is an upper bound (early stopping only helps).",
        "",
        "| Study | Hours (cap 5) | of which Manifold | Hours (cap 10) | of which Manifold |",
        "| --- | --- | --- | --- | --- |",
    ]
    for study in studies.STUDIES:
        h5, m5 = per_study[5][study]
        h10, m10 = per_study[10][study]
        mark = " ~" if any(s == study for s, _ in timings.approximated) else ""
        lines.append(f"| {study}{mark} | {h5:.1f} | {m5:.1f} | "
                     f"{h10:.1f} | {m10:.1f} |")
    lines.append(f"| **total** | **{totals[5]:.1f}** |  | "
                 f"**{totals[10]:.1f}** |  |")
    lines.append("")
    if timings.approximated:
        lines.append("`~` = no direct measurement for "
                     + ", ".join(sorted(f"{s}/{o}" for s, o in timings.approximated))
                     + " — scaled from the nearest same-architecture "
                       "measurement by analytic per-step FLOPs.")
        lines.append("")
    if timings.missing:
        lines.append("**WARNING — excluded from the totals (no measurement "
                     "anywhere):** " + ", ".join(sorted(timings.missing))
                     + ". Run scripts/smoke.py for these before trusting "
                       "the totals.")
        lines.append("")

    # PLAN's rules: cap 10 only if it fits; else cap 5; else cuts, in order.
    w = args.window_hours
    lines.append("## Verdict and cut order (PLAN \"If the budget is tight\")")
    lines.append("")
    if totals[10] <= w:
        lines.append(f"* Cap 10 fits ({totals[10]:.1f} h <= {w:.0f} h): "
                     f"**run Manifold Muon at dual-step cap 10**.")
    elif totals[5] <= w:
        lines.append(f"* Cap 10 does NOT fit ({totals[10]:.1f} h > {w:.0f} h); "
                     f"cap 5 does ({totals[5]:.1f} h): **keep the default "
                     f"cap 5**.")
    else:
        lines.append(f"* Neither cap fits (cap 5: {totals[5]:.1f} h > "
                     f"{w:.0f} h). Cuts, in the plan's order:")
        running = totals[5]
        _, moe_man = per_study[5]["moe"]
        running -= moe_man
        fit = "FITS" if running <= w else "still does not fit"
        lines.append(f"  1. Drop Manifold Muon from the MoE study "
                     f"(-{moe_man:.1f} h -> {running:.1f} h): {fit}. "
                     f"Keep the MoE itself — it is Study B's only comparison.")
        if running > w:
            _, m512 = per_study[5]["mlp512"]
            running -= m512
            fit = "FITS" if running <= w else "still does not fit"
            lines.append(f"  2. Then drop Manifold Muon from MLP-512 "
                         f"(-{m512:.1f} h -> {running:.1f} h): {fit}.")
        if running > w:
            lines.append("  3. Remaining options are STUDY-CHANGING decisions "
                         "for the lead, not applied here: raise the MLP batch "
                         "to 1,024 (decide before any run), and as a last "
                         "resort drop MLP-256. Do not drop the MoE.")
    lines.append("")
    lines.append("Faithful-settings reruns are included above (Manifold's at "
                 "cap 100 — the dominant single run; PLAN schedules it at "
                 "MLP-512 only if time remains).")
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
