# Compute estimate — measured s/step x planned job list

Generated 2026-11-17T20:37:40.014862+00:00 from `hardware-results.csv`; window 30 h.

Scenario: `studies.planned_jobs` (no accepted coordinate-descent change, no edge trials, half-LR not promoted) — mlp128: 101, mlp256: 161, mlp512: 161, dense: 101, moe: 94 = 618 runs. Manifold msign cost is an upper bound (early stopping only helps).

| Study | Hours (cap 5) | of which Manifold | Hours (cap 10) | of which Manifold |
| --- | --- | --- | --- | --- |
| mlp128 | 0.5 | 0.3 | 0.6 | 0.4 |
| mlp256 | 1.7 | 0.9 | 2.2 | 1.4 |
| mlp512 | 3.6 | 2.4 | 4.7 | 3.5 |
| dense | 7.1 | 1.7 | 7.4 | 2.0 |
| moe | 80.3 | 17.0 | 85.4 | 22.1 |
| **total** | **93.2** |  | **100.3** |  |

## Verdict and cut order (PLAN "If the budget is tight")

* Neither cap fits (cap 5: 93.2 h > 30 h). Cuts, in the plan's order:
  1. Drop Manifold Muon from the MoE study (-17.0 h -> 76.2 h): still does not fit. Keep the MoE itself — it is Study B's only comparison.
  2. Then drop Manifold Muon from MLP-512 (-2.4 h -> 73.8 h): still does not fit.
  3. Remaining options are STUDY-CHANGING decisions for the lead, not applied here: raise the MLP batch to 1,024 (decide before any run), and as a last resort drop MLP-256. Do not drop the MoE.

Faithful-settings reruns are included above (Manifold's at cap 100 — the dominant single run; PLAN schedules it at MLP-512 only if time remains).
