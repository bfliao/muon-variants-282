"""Tests for muonarch/analysis.py (design.md §3.11).

All tests run on a SYNTHETIC world: fake result JSONs under
results/<study>/*.json and fake JSONL logs under logs/<study>/*.jsonl,
generated into a tmp directory.  No training code is imported.

Synthetic world contents (see make_world):
  - study "dense" (Study B) tuned Phase-II style: stage_a (6 lr trials),
    stage_b (6 lr x wd trials), fresh seeds 1-3, AdamW budget curve
    (3 branches + 1x + constant-2x per seed, half-LR 2x promoted to all
    seeds), one diverged trial and one ">best + 0.5 nats" trial (muonall);
  - study "mlp128" (Study A) tuned Phase-I style: coordinate-descent trials
    (phase "cd" with a cd_sweep indicator) reproducing the incumbent-label
    case of the runner review's probe D: a pass-1 lr sweep at wd=0.01, a wd
    sweep whose winning cell (wd=0.03 at the centre lr) IS the deduped
    centre of the pass-2 lr sweep (so it keeps its "wd" label), and the
    pass-2 lr sweep's remaining 5 cells at wd=0.03; plus fresh seeds 1-3,
    AdamW budget curve, half-LR 2x on seed 1 only;
  - study "moe": fresh seeds only (so missing-data paths are exercised);
  - AdamW losses lie exactly on a known curve L(D) = a*D^-b + g per study,
    and every other optimizer's final loss is planted at curve(s * D_1x),
    so the fitted/inverted token speedup must recover the planted s.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from muonarch import analysis as an

# ---------------------------------------------------------------- world spec

STUDY_SPEC = {
    "dense": dict(arch="dense", width=None, base=2441, batch=32, seq=256, unit="tokens"),
    "mlp128": dict(arch="mlp", width=128, base=3520, batch=256, seq=None, unit="images"),
    "moe": dict(arch="moe", width=None, base=2441, batch=32, seq=256, unit="tokens"),
}
D1 = {
    "dense": 2441 * 32 * 256,
    "mlp128": 3520 * 256,
    "moe": 2441 * 32 * 256,
}
# L(D) = alpha * D^-beta + gamma (AdamW data-budget curve per study)
CURVES = {"dense": (2.0, 0.3, 3.0), "mlp128": (1.5, 0.25, 1.2)}
# planted token speedups: L_opt = curve(s * D_1x)
PLANTED = {
    "dense": {"muon": 1.3, "moonlight": 1.15, "manifold": 0.9, "muonall": 1.0},
    "mlp128": {"muon": 1.1, "moonlight": 1.05, "manifold": 0.8, "muonall": 0.95},
}
# total-FLOPs-per-unit rates (model, optimizer) per study x optimizer
MODEL_RATE = {"dense": 10_000.0, "mlp128": 50_000.0, "moe": 14_000.0}
OPT_RATE = {
    "adamw": 0.01,      # fraction of the model rate
    "muon": 0.30,
    "moonlight": 0.30,
    "manifold": 3.0,
    "muonall": 0.35,
}
BEST_LR = {"adamw": 0.1, "muon": 0.02, "moonlight": 0.1, "manifold": 0.1, "muonall": 0.02}
# stage-A loss offsets above the optimizer's best, index 2 is the grid winner
STAGE_A_OFF = {
    "dense": [0.09, 0.015, 0.0, 0.012, 0.035, 0.30],
    "mlp128": [0.05, 0.010, 0.0, 0.011, 0.02, 0.20],
    "moe": [0.09, 0.015, 0.0, 0.012, 0.035, 0.30],
}
SEED_EPS = {1: 0.0, 2: 2e-5, 3: -2e-5}
INIT_LOSS = {"dense": math.log(4096), "mlp128": math.log(10), "moe": math.log(4096)}
MOE_FRESH = {"adamw": 3.05, "muon": 3.02, "moonlight": 3.06, "manifold": 3.10, "muonall": 3.05}


def curve(study: str, D: float) -> float:
    a, b, g = CURVES[study]
    return a * D ** -b + g


def total_rate(study: str, opt: str) -> float:
    """Total training FLOPs per data unit (model + optimizer) for a run."""
    return MODEL_RATE[study] * (1.0 + OPT_RATE[opt])


def _cfg(study, opt, phase, seed, lr, wd, mult, **extra):
    s = STUDY_SPEC[study]
    cfg = {
        "schema_version": 1,
        "study": study,
        "phase": phase,
        "arch": s["arch"],
        "width": s["width"],
        "optimizer": opt,
        "lr": lr,
        "wd": wd,
        "seed": seed,
        "budget_multiplier": mult,
        "budget_base_steps": s["base"],
        "budget_unit": s["unit"],
        "batch": s["batch"],
        "seq_len": s["seq"],
        "precision": "fp32",
    }
    cfg.update(extra)
    return cfg


def _write_result(root: Path, cfg: dict, loss, status="completed",
                  diverged_at=None, edge_fired=False):
    study, opt = cfg["study"], cfg["optimizer"]
    h = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
    d_run = cfg["budget_multiplier"] * D1[study]
    fm = MODEL_RATE[study] * d_run
    fo = MODEL_RATE[study] * OPT_RATE[opt] * d_run
    res = {
        "hash": h,
        "schema_version": 1,
        "config": cfg,
        "status": status,
        "diverged_at_step": diverged_at,
        "final_val_loss": loss,
        "final_val_acc": None if study != "mlp128" else 0.5,
        "final_test_loss": loss,
        "final_test_acc": None if study != "mlp128" else 0.5,
        "train_val_gap": 0.01,
        "flops": {"model": fm, "optimizer": fo, "total": fm + fo},
        "dual_steps_mean": 4.2 if opt == "manifold" else None,
        "msign_calls_mean": 6.0 if opt == "manifold" else None,
        "edge_rule_fired": edge_fired,
        "device": "cpu",
        "precision": "fp32",
        "wall_clock_s": 1.0,
    }
    out = root / "results" / study / f"{h}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    return h, res


def _write_log(root: Path, cfg: dict, h: str, final_loss: float):
    study, opt = cfg["study"], cfg["optimizer"]
    s = STUDY_SPEC[study]
    total = round(cfg["budget_multiplier"] * s["base"])
    unit_per_step = s["batch"] * (s["seq"] or 1)
    init = INIT_LOSS[study]
    lines = [{"t": "start", "hash": h, "config": cfg}]
    for k in range(1, 9):
        step = round(k * total / 8)
        frac = step / total
        val = final_loss + (init - final_loss) * math.exp(-5 * frac)
        rec = {
            "t": "eval", "step": step, "images_or_tokens": step * unit_per_step,
            "lr": cfg["lr"], "train_loss": val - 0.01, "val_loss": val,
            "grad_norm": 1.0,
        }
        if s["arch"] in ("dense", "moe"):
            rec["max_attn_logit"] = 4.0 + 2.0 * frac + (0.5 if opt != "adamw" else 0.0)
        if s["arch"] == "moe":
            rec["expert_entropy"] = 2.2 - (0.3 if opt != "adamw" else 0.1) * frac
            rec["expert_share_min"] = 0.02
            rec["expert_share_max"] = 0.12
            rec["aux_loss"] = 0.01
        lines.append(rec)
    lines.append({
        "t": "final", "val_loss": final_loss, "test_loss": final_loss,
        "test_acc": 0.5,
        "spectral_norms": {"layer0.weight": 1.0 + 0.1 * len(opt) % 1,
                           "layer1.weight": 1.5, "layer2.weight": 2.0},
        "flops": {"model": 1.0, "optimizer": 1.0, "total": 2.0},
    })
    out = root / "logs" / study / f"{h}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(x) for x in lines) + "\n")


def make_world(root: Path) -> None:
    for study in ("dense", "mlp128"):
        d1 = D1[study]
        # ---- AdamW budget curve, seeds 1-3 (exactly on the planted curve)
        for seed in (1, 2, 3):
            for m in (0.125, 0.25, 0.5):
                cfg = _cfg(study, "adamw", "budget_branch", seed, BEST_LR["adamw"], 0.1, m)
                _write_result(root, cfg, curve(study, m * d1))
            # constant run to 2x: its own decay is the branched 2x point;
            # handicapped slightly on dense so the half-LR run wins there
            handicap = 3e-3 if study == "dense" else 0.0
            cfg = _cfg(study, "adamw", "budget_constant", seed, BEST_LR["adamw"], 0.1, 2.0)
            _write_result(root, cfg, curve(study, 2 * d1) + handicap)
        # half-LR 2x runs (seed-1-first promotion rule): on dense the seed-1
        # half-LR run beat the branched 2x, so it was promoted to every curve
        # seed; on mlp128 it lost on seed 1, so only that one run exists and
        # the branched 2x stands everywhere.
        if study == "dense":
            for seed in (1, 2, 3):
                cfg = _cfg(study, "adamw", "budget_halflr", seed,
                           BEST_LR["adamw"] / 2, 0.1, 2.0)
                _write_result(root, cfg, curve(study, 2 * d1))
        else:
            cfg = _cfg(study, "adamw", "budget_halflr", 1, BEST_LR["adamw"] / 2, 0.1, 2.0)
            _write_result(root, cfg, curve(study, 2 * d1) + 5e-3)

        # ---- fresh seeds 1-3 for every optimizer (adamw's double as 1x points)
        for opt in an.OPTIMIZERS:
            for seed in (1, 2, 3):
                s = PLANTED[study].get(opt, 1.0)
                loss = curve(study, s * d1) + SEED_EPS[seed]
                cfg = _cfg(study, opt, "fresh_seed", seed, BEST_LR[opt],
                           None if opt == "manifold" else 0.01, 1.0)
                h, _ = _write_result(root, cfg, loss)
                _write_log(root, cfg, h, loss)

        # ---- tuning trials, seed 0
        for opt in an.OPTIMIZERS:
            best = curve(study, PLANTED[study].get(opt, 1.0) * d1)
            lrs = [BEST_LR[opt] * 2 ** (i - 2) for i in range(6)]
            if study == "dense":  # Phase-II style: stage A + stage B
                for i, (lr, off) in enumerate(zip(lrs, STAGE_A_OFF[study])):
                    cfg = _cfg(study, opt, "stage_a", 0, lr,
                               None if opt == "manifold" else 0.01, 1.0)
                    if opt == "muonall" and i == 5:
                        _write_result(root, cfg, None, status="diverged",
                                      diverged_at=500)
                    else:
                        edge = opt == "adamw" and i == 0
                        _write_result(root, cfg, best + off, edge_fired=edge)
                if opt == "manifold":
                    bgrid = [(lrs[2] * 2 ** k, None)
                             for k in (-0.75, -0.5, -0.25, 0.25, 0.5, 0.75)]
                else:
                    bgrid = [(lr, wd) for lr in (lrs[2], lrs[3])
                             for wd in (0.0, 0.003, 0.03)]
                for j, (lr, wd) in enumerate(bgrid):
                    off = 0.002 * (j + 1)
                    if opt == "muonall" and j == 5:
                        off = 0.6  # > best + 0.5 nats: diverged at analysis time
                    cfg = _cfg(study, opt, "stage_b", 0, lr, wd, 1.0)
                    _write_result(root, cfg, best + off)
            else:  # mlp128, Phase-I style: coordinate descent ("cd") trials
                if opt == "manifold":  # no wd: a single lr sweep
                    for lr, off in zip(lrs, STAGE_A_OFF[study]):
                        cfg = _cfg(study, opt, "cd", 0, lr, None, 1.0,
                                   cd_sweep="lr")
                        _write_result(root, cfg, best + off)
                else:  # probe-D shape: two lr-sweep passes around a wd acceptance
                    # pass-1 lr sweep at the pre-acceptance wd=0.01 (worse context)
                    for lr, off in zip(lrs, STAGE_A_OFF[study]):
                        cfg = _cfg(study, opt, "cd", 0, lr, 0.01, 1.0,
                                   cd_sweep="lr")
                        _write_result(root, cfg, best + off + 0.02)
                    # wd sweep at the centre lr; wd=0.03 wins, and that cell
                    # IS the pass-2 lr sweep's centre: config dedup keeps its
                    # FIRST label, cd_sweep=="wd"
                    for wd, off in ((0.0, 0.004), (0.003, 0.002), (0.03, 0.0)):
                        cfg = _cfg(study, opt, "cd", 0, lrs[2], wd, 1.0,
                                   cd_sweep="wd")
                        _write_result(root, cfg, best + off)
                    # pass-2 lr sweep at the accepted wd=0.03: the centre cell
                    # is deduped away, only the other 5 cells are written
                    for i, (lr, off) in enumerate(zip(lrs, STAGE_A_OFF[study])):
                        if i == 2:
                            continue
                        cfg = _cfg(study, opt, "cd", 0, lr, 0.03, 1.0,
                                   cd_sweep="lr")
                        _write_result(root, cfg, best + off)

    # ---- edge-extension trials (for edge_rule_report), both labelings:
    # Phase-I lr-edge extension: phase "cd" with the logged-only
    # edge_trial=true field (the phase alone would miss it)
    cfg = _cfg("mlp128", "muon", "cd", 0, BEST_LR["muon"] * 2 ** 4, 0.03, 1.0,
               cd_sweep="lr", edge_trial=True)
    _write_result(root, cfg, curve("mlp128", PLANTED["mlp128"]["muon"] * D1["mlp128"]) + 0.4)
    # Phase-II edge trial from before the edge_trial field existed:
    # phase "edge", no edge_trial field -> counted via the phase fallback
    cfg = _cfg("dense", "muon", "edge", 0, BEST_LR["muon"] * 2 ** 4, 0.01, 1.0)
    _write_result(root, cfg, curve("dense", PLANTED["dense"]["muon"] * D1["dense"]) + 0.4)

    # ---- moe: fresh seeds only (no budget curve, no tuning on disk)
    for opt in an.OPTIMIZERS:
        for seed in (1, 2, 3):
            loss = MOE_FRESH[opt] + SEED_EPS[seed]
            cfg = _cfg("moe", opt, "fresh_seed", seed, BEST_LR[opt],
                       None if opt == "manifold" else 0.01, 1.0)
            h, _ = _write_result(root, cfg, loss)
            _write_log(root, cfg, h, loss)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("world")
    make_world(root)
    return root


@pytest.fixture(scope="module")
def df(world):
    return an.load_results(world)


@pytest.fixture(scope="module")
def logs(world):
    return an.load_logs(world)


# ---------------------------------------------------------------- loading

def test_load_results_flattens(df):
    for col in ("study", "optimizer", "phase", "seed", "lr", "wd",
                "budget_multiplier", "status", "final_val_loss",
                "flops_total", "flops_model", "flops_optimizer"):
        assert col in df.columns, col
    assert set(df["study"]) == {"dense", "mlp128", "moe"}
    assert set(df["optimizer"]) == set(an.OPTIMIZERS)
    div = df[df["status"] == "diverged"]
    assert len(div) == 1
    assert div.iloc[0]["optimizer"] == "muonall"
    assert np.isnan(div.iloc[0]["final_val_loss"])
    # row count: dense 28+15*3+16 (adamw 13 curve/branch/halflr + 15 trials&fresh)
    assert len(df) == len(list((Path(df.attrs["root"]) / "results").rglob("*.json")))


def test_load_logs(logs):
    assert {"study", "optimizer", "seed", "step", "val_loss",
            "images_or_tokens", "t"} <= set(logs.columns)
    ev = logs[logs["t"] == "eval"]
    assert (ev.groupby(["study", "optimizer", "seed"])["step"].count() == 8).all()
    moe = ev[ev["study"] == "moe"]
    assert moe["expert_entropy"].notna().all()
    fin = logs[logs["t"] == "final"]
    assert fin["spectral_norms"].notna().all()


# ---------------------------------------------------------------- curve fit

def test_fit_recovers_curve_and_speedup():
    a, b, g = 2.0, 0.3, 3.0
    d1 = D1["dense"]
    pts = [(m * d1, a * (m * d1) ** -b + g) for m in (0.125, 0.25, 0.5, 1.0, 2.0)]
    fit = an.fit_budget_curve(pts)
    for D, L in pts:
        assert abs(an.FitResult(fit.alpha, fit.beta, fit.gamma).alpha * D ** -fit.beta
                   + fit.gamma - L) < 1e-6
    # planted speedup 1.3 recovered within 2%
    L_opt = a * (1.3 * d1) ** -b + g
    assert abs(an.invert_curve(fit, L_opt) / (1.3 * d1) - 1) < 0.02


def test_invert_curve_inf():
    fit = an.FitResult(alpha=2.0, beta=0.3, gamma=3.0)
    assert math.isinf(an.invert_curve(fit, 3.0))
    assert math.isinf(an.invert_curve(fit, 2.5))
    assert an.invert_curve(fit, 3.1) < float("inf")
    # degenerate fits: alpha/beta <= 0, or beta pinned to a scan boundary
    assert math.isinf(an.invert_curve(an.FitResult(-1.0, 0.3, 3.0), 3.1))
    assert math.isinf(an.invert_curve(an.FitResult(2.0, an.BETA_SCAN_LO, 3.0), 3.1))
    assert math.isinf(an.invert_curve(an.FitResult(2.0, an.BETA_SCAN_HI, 3.0), 3.1))
    # near-boundary beta would overflow the float power: inf, not a raise
    assert math.isinf(an.invert_curve(an.FitResult(2.0, 1.2e-3, 3.0), 3.1))


def test_fit_flat_points_inverts_to_inf():
    # near-constant losses: no usable curvature; beta lands on a scan
    # boundary and invert_curve treats the fit as degenerate on BOTH sides
    rng = np.random.default_rng(0)
    d1 = D1["dense"]
    pts = [(m * d1, 3.0 + rng.normal(0, 1e-4)) for m in (0.125, 0.25, 0.5, 1.0, 2.0)]
    fit = an.fit_budget_curve(pts)
    for loss in (2.99, 3.01):
        assert math.isinf(an.invert_curve(fit, loss))  # must not raise


# ------------------------------------------------- point assembly / promotion

def test_curve_points_halflr_promotion(df):
    # dense seed 1: half-LR 2x beats the branched 2x -> min picked (half-LR)
    pts = dict(an.curve_points("dense", 1, df))
    d1 = D1["dense"]
    assert len(pts) == 5
    assert pts[2 * d1] == pytest.approx(curve("dense", 2 * d1), abs=1e-12)
    # dense seed 2: promoted half-LR run beats the handicapped branched 2x
    pts2 = dict(an.curve_points("dense", 2, df))
    assert pts2[2 * d1] == pytest.approx(curve("dense", 2 * d1), abs=1e-12)
    # mlp128 seed 1: half-LR exists but is WORSE -> branched 2x picked
    ptsm = dict(an.curve_points("mlp128", 1, df))
    dm = D1["mlp128"]
    assert ptsm[2 * dm] == pytest.approx(curve("mlp128", 2 * dm), abs=1e-12)
    # mlp128 seed 2: no half-LR run exists -> branched 2x stands
    ptsm2 = dict(an.curve_points("mlp128", 2, df))
    assert ptsm2[2 * dm] == pytest.approx(curve("mlp128", 2 * dm), abs=1e-12)
    # 1x point is the fresh-seed run
    assert pts[d1] == pytest.approx(curve("dense", d1), abs=1e-12)


def test_curve_points_warns_on_incomplete_assembly(df):
    # a missing branch warns (log, not raise) and fits what exists
    drop = ((df["study"] == "mlp128") & (df["phase"] == "budget_branch")
            & (df["budget_multiplier"] == 0.125) & (df["seed"] == 1))
    with pytest.warns(UserWarning, match="4 points"):
        pts = an.curve_points("mlp128", 1, df[~drop])
    assert len(pts) == 4


# ---------------------------------------------------------------- speedups

def test_speedup_range_over_seeds(df):
    out = an.token_speedup_range("dense", df)
    assert set(PLANTED["dense"]) <= set(out.index)
    for opt, s in PLANTED["dense"].items():
        row = out.loc[opt]
        m = np.asarray(row["matrix"], dtype=float)
        assert m.shape == (3, 3)  # 3 fitted curves x 3 optimizer seeds
        assert np.all(np.abs(m / s - 1) < 0.02), (opt, m)
        assert row["speedup_min"] == pytest.approx(m.min())
        assert row["speedup_max"] == pytest.approx(m.max())
        assert row["speedup_min"] <= row["speedup_max"]
    # manifold is slower than AdamW: entire range below 1
    assert out.loc["manifold", "speedup_max"] < 1.0


def test_flop_speedup_convention(df):
    out = an.flop_speedup_range("dense", df)
    for opt, s in PLANTED["dense"].items():
        expected = s * total_rate("dense", "adamw") / total_rate("dense", opt)
        m = np.asarray(out.loc[opt, "matrix"], dtype=float)
        assert m.shape == (3, 3)
        assert np.all(np.abs(m / expected - 1) < 0.02), (opt, m, expected)
    # Manifold pays ~4x total FLOPs: far below break-even
    assert out.loc["manifold", "speedup_max"] < 0.5


def test_speedup_missing_curve_is_empty(df):
    out = an.token_speedup_range("moe", df)  # no budget curve on disk
    assert len(out) == 0


# ---------------------------------------------------------------- basin width

def test_basin_width(df):
    # dense (stage_a): offsets [0.09, 0.015, 0, 0.012, 0.035, 0.30], tol 0.02
    # nats absolute -> lrs i=1..3 within, width = log2(lr3/lr1) = 2 factors of 2
    w = an.lr_basin_width("dense", "adamw", df, tol=0.02)
    assert w == pytest.approx(2.0)


def test_basin_width_mlp_relative_from_cd(df):
    # mlp128 is Phase-I tuned: its lr sweep lives in phase=="cd" rows with
    # cd_sweep=="lr".  best = curve(1.0*D1) ~ 1.2486; 1% relative tol = 0.0125
    # offsets [0.05, 0.010, 0, 0.011, 0.02, 0.20] -> i=1..3 within -> width 2
    w = an.lr_basin_width("mlp128", "adamw", df, tol=0.01)
    assert w == pytest.approx(2.0)


def test_basin_width_contiguous_around_best():
    # non-contiguous within-tol set {lr=1, lr=32} with interior cells +0.5
    # nats out: the basin is only the best cell itself -> width 0.0, not 5.0
    rows = [dict(study="dense", optimizer="adamw", phase="stage_a", seed=0,
                 lr=2 ** k, final_val_loss=L, status="completed")
            for k, L in zip(range(6), [3.00, 3.50, 3.50, 3.50, 3.50, 3.01])]
    w = an.lr_basin_width("dense", "adamw", pd.DataFrame(rows), tol=0.02)
    assert w == pytest.approx(0.0)
    # contiguous run 2,4,8 within tol -> log2(8/2) = 2
    rows2 = [dict(study="dense", optimizer="adamw", phase="stage_a", seed=0,
                  lr=2 ** k, final_val_loss=3.0 + (0.0 if 1 <= k <= 3 else 0.5),
                  status="completed") for k in range(6)]
    w2 = an.lr_basin_width("dense", "adamw", pd.DataFrame(rows2), tol=0.02)
    assert w2 == pytest.approx(2.0)


def test_lr_sweep_rows_recovers_deduped_centre(df):
    # probe D (runner review): the pass-2 lr sweep's centre cell was first
    # run inside the wd sweep, so dedup left it labeled cd_sweep=="wd".  The
    # sweep selection must (a) re-admit it via context matching and (b) drop
    # the pass-1 lr-labeled cells from the pre-acceptance wd=0.01 context.
    rows = an._lr_sweep_rows(df, "mlp128", "adamw")
    assert len(rows) == 6
    assert set(np.round(rows["wd"].astype(float), 12)) == {0.03}
    expected_lrs = [BEST_LR["adamw"] * 2 ** (i - 2) for i in range(6)]
    assert sorted(rows["lr"]) == pytest.approx(expected_lrs)
    centre = rows[rows["lr"] == BEST_LR["adamw"]]
    assert len(centre) == 1
    assert centre.iloc[0]["cd_sweep"] == "wd"  # the deduped centre, re-admitted
    assert (rows[rows["lr"] != BEST_LR["adamw"]]["cd_sweep"] == "lr").all()


def test_basin_width_cd_fallback_grouping():
    # Phase-I rows labeled "cd" WITHOUT a cd_sweep indicator: the lr sweep is
    # recovered as the group (by other tuned hypers) containing the best row
    rows = [dict(study="dense", optimizer="adamw", phase="cd", seed=0, wd=0.1,
                 lr=2 ** k, final_val_loss=3.0 + (0.0 if 1 <= k <= 3 else 0.5),
                 status="completed") for k in range(6)]
    # a wd-sweep cell at another wd whose far-out lr must NOT join the sweep
    rows.append(dict(study="dense", optimizer="adamw", phase="cd", seed=0,
                     wd=0.3, lr=2 ** 5, final_val_loss=3.001, status="completed"))
    w = an.lr_basin_width("dense", "adamw", pd.DataFrame(rows), tol=0.02)
    assert w == pytest.approx(2.0)


# ---------------------------------------------------------------- divergence

def test_divergence_rate(df):
    # muonall dense: 15 trials total (6 stage_a incl. 1 diverged, 6 stage_b
    # incl. 1 at best+0.6, 3 fresh); both criteria counted
    r = an.divergence_rate("dense", "muonall", df)
    assert r == pytest.approx(2 / 15)
    assert an.divergence_rate("dense", "adamw", df) == 0.0


def test_divergence_rate_excludes_budget_curve():
    # AdamW's own short decay branches end ~1 nat above its 2x loss BY DESIGN;
    # budget-curve phases must not enter numerator or denominator
    def res(phase, seed, mult, loss):
        return dict(study="dense", optimizer="adamw", phase=phase, seed=seed,
                    lr=0.1, wd=0.1, budget_multiplier=mult, status="completed",
                    final_val_loss=loss)
    rows = []
    for seed in (1, 2, 3):
        rows += [res("budget_branch", seed, 0.125, 4.63),
                 res("budget_branch", seed, 0.25, 4.25),
                 res("budget_branch", seed, 0.5, 3.95),
                 res("fresh_seed", seed, 1.0, 3.70),
                 res("budget_constant", seed, 2.0, 3.45)]
    assert an.divergence_rate("dense", "adamw", pd.DataFrame(rows)) == 0.0


# ---------------------------------------------------------------- seed rule

def test_seed_rule_sign_and_2std():
    # all same sign, mean > 2*max(std) -> True
    assert an.seed_rule(np.array([-0.05, -0.06, -0.055]), (0.01, 0.02)) is True
    assert an.seed_rule(np.array([0.05, 0.06, 0.055]), (0.01, 0.02)) is True
    # mixed signs -> False even with a big mean
    assert an.seed_rule(np.array([0.5, -0.01, 0.6]), (0.01, 0.01)) is False
    # same sign but mean below 2*max(std) -> False
    assert an.seed_rule(np.array([-0.05, -0.06, -0.055]), (0.01, 0.03)) is False
    # zero diff -> False
    assert an.seed_rule(np.array([0.0, 0.0, 0.0]), (0.0, 0.0)) is False


# ---------------------------------------------------------------- H3

def test_h3_interaction_planted():
    rows = []
    for seed, (la_d, la_m) in enumerate(zip([3.00, 3.01, 3.02], [3.30, 3.31, 3.32]), 1):
        rows.append(dict(study="dense", optimizer="adamw", phase="fresh_seed",
                         seed=seed, final_val_loss=la_d, status="completed"))
        rows.append(dict(study="moe", optimizer="adamw", phase="fresh_seed",
                         seed=seed, final_val_loss=la_m, status="completed"))
        # muon: gap -0.02 on dense, -0.08 on moe -> interaction -0.06, tiny std
        rows.append(dict(study="dense", optimizer="muon", phase="fresh_seed",
                         seed=seed, final_val_loss=la_d - 0.02, status="completed"))
        rows.append(dict(study="moe", optimizer="muon", phase="fresh_seed",
                         seed=seed, final_val_loss=la_m - 0.08 + [0, 1e-3, -1e-3][seed - 1],
                         status="completed"))
        # moonlight: same gap on both archs up to noise -> no interaction
        rows.append(dict(study="dense", optimizer="moonlight", phase="fresh_seed",
                         seed=seed, final_val_loss=la_d - 0.02, status="completed"))
        rows.append(dict(study="moe", optimizer="moonlight", phase="fresh_seed",
                         seed=seed, final_val_loss=la_m - 0.02 + [0.01, -0.01, 0.0][seed - 1],
                         status="completed"))
    out = an.h3_interaction(pd.DataFrame(rows))
    muon = out.loc["muon"]
    assert muon["mean_gap_diff"] == pytest.approx(-0.06, abs=1e-6)
    assert bool(muon["real"]) is True
    moonlight = out.loc["moonlight"]
    assert abs(moonlight["mean_gap_diff"]) < 0.01
    assert bool(moonlight["real"]) is False


def test_h3_single_paired_seed_never_real():
    # one paired seed gives no spread estimate: never "real", however big
    rows = []
    for study, la in (("dense", 3.00), ("moe", 3.30)):
        rows.append(dict(study=study, optimizer="adamw", phase="fresh_seed",
                         seed=1, final_val_loss=la, status="completed"))
        rows.append(dict(study=study, optimizer="muon", phase="fresh_seed",
                         seed=1, final_val_loss=la - (0.02 if study == "dense" else 0.10),
                         status="completed"))
    out = an.h3_interaction(pd.DataFrame(rows))
    assert int(out.loc["muon", "n_seeds"]) == 1
    assert bool(out.loc["muon", "real"]) is False


def test_h3_on_world(df):
    out = an.h3_interaction(df)
    assert set(out.index) == {"muon", "moonlight", "manifold", "muonall"}
    assert {"mean_gap_diff", "std_gap_diff", "real"} <= set(out.columns)


# ------------------------------------------------- null results / edge / verdicts

def test_null_result_rows(df):
    out = an.null_result_rows(df)
    assert {"study", "optimizer", "mean_diff", "seed_spread", "verdict"} <= set(out.columns)
    assert (out["verdict"] == "indistinguishable after equal tuning").all()
    # moe: muonall and adamw are planted identical -> must appear as a null row
    assert len(out[(out["study"] == "moe") & (out["optimizer"] == "muonall")]) == 1


def test_edge_rule_report(df):
    rep = an.edge_rule_report(df)
    assert "edge rule" in rep.lower()
    assert "fired 1 time" in rep.lower()          # edge_rule_fired: dense adamw
    assert "fired — dense/adamw: 1" in rep
    # both edge-trial labelings counted, per optimizer x study:
    # cd-phase row with edge_trial=true AND legacy phase=="edge" row
    assert "2 extra edge trial(s)" in rep
    assert "edge trials — mlp128/muon: 1" in rep  # phase "cd", edge_trial=True
    assert "edge trials — dense/muon: 1" in rep   # phase "edge", no field


def test_hypothesis_verdicts(df, logs):
    out = an.hypothesis_verdicts(df, logs=logs)
    assert list(out["hypothesis"]) == ["H1", "H2a", "H2b", "H3", "H4", "H5", "H6", "H7"]
    assert set(out["verdict"]) <= {"supported", "falsified", "indistinguishable"}
    # H1: best dense variant is muon with planted speedup 1.3 in [1.0, 1.4]
    assert out[out["hypothesis"] == "H1"].iloc[0]["verdict"] == "supported"
    # H4 needs mlp512 (absent) -> indistinguishable, not a crash
    assert out[out["hypothesis"] == "H4"].iloc[0]["verdict"] == "indistinguishable"
    # every row carries the heuristic caveat
    assert out["caveat"].str.contains("heuristic").all()


# ---------------------------------------------------------------- figures/table

def test_all_figures_render(df, logs, tmp_path):
    outdir = tmp_path / "figures"
    paths = [
        an.figure_1(df, outdir),
        an.figure_2(df, outdir),
        an.figure_3(df, outdir, logs=logs),
        an.figure_4(df, outdir),
        an.figure_5(df, outdir),
        an.figure_6(df, outdir),
        an.figure_7(df, outdir),
        an.figure_8(df, outdir, logs=logs),
    ]
    for i, p in enumerate(paths, 1):
        assert p.exists() and p.stat().st_size > 0, p
        assert p.suffix == ".png"
        assert p.name.startswith(f"fig{i}"), p.name
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_figures_render_without_logs(df, tmp_path):
    # figures 3 and 8 must still produce a valid PNG without log data
    outdir = tmp_path / "figures"
    for fn in (an.figure_3, an.figure_8):
        p = fn(df, outdir)
        assert p.exists() and p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_main_table_mean_std(df, tmp_path):
    outdir = tmp_path / "tables"
    p = an.main_table(df, outdir)
    assert p == outdir / "main.md"
    text = p.read_text()
    assert "Study A" in text and "Study B" in text
    # Study A and Study B are separate sections, no cross-study ranking
    assert text.index("Study A") < text.index("Study B")
    assert "heuristic" in text.lower()
    assert "±" in text
    # per-metric columns present
    for col in ("token speedup", "FLOP speedup", "basin", "divergence"):
        assert col.lower() in text.lower()
    # dense winner muon passes the seed rule (planted gap >> spread) -> bold
    assert "**muon**" in text
    # moe has no budget curve: speedup cells degrade gracefully, not crash
    assert "moe" in text


def test_inf_speedup_robust_figure_and_table(tmp_path):
    # an optimizer whose loss lies BELOW the fitted AdamW asymptote gives inf
    # speedup cells; figure_1 and main_table must render, not crash
    def res(opt, phase, seed, mult, loss, lr=0.1):
        return dict(study="dense", optimizer=opt, phase=phase, seed=seed,
                    lr=lr, wd=0.1, budget_multiplier=mult,
                    budget_base_steps=2441, batch=32, seq_len=256,
                    budget_unit="tokens", status="completed",
                    final_val_loss=loss, flops_total=1.0, flops_model=1.0,
                    flops_optimizer=0.0)
    rows = []
    d1 = D1["dense"]
    for seed in (1, 2):
        for m in (0.125, 0.25, 0.5):
            rows.append(res("adamw", "budget_branch", seed, m, curve("dense", m * d1)))
        rows.append(res("adamw", "fresh_seed", seed, 1.0, curve("dense", d1)))
        rows.append(res("adamw", "budget_constant", seed, 2.0, curve("dense", 2 * d1)))
        rows.append(res("muon", "fresh_seed", seed, 1.0, 2.9, lr=0.02))  # < gamma=3.0
    mini = pd.DataFrame(rows)
    out = an.token_speedup_range("dense", mini)
    assert math.isinf(out.loc["muon", "speedup_max"])
    p = an.figure_1(mini, tmp_path / "figures")
    assert p.exists() and p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    t = an.main_table(mini, tmp_path / "tables")
    assert "inf" in t.read_text()


# ---------------------------------------------------- two tuning tracks (#12)

def _two_track_df():
    """mlp256/adamw with BOTH a cd lr sweep (headline, best lr 2e-3) and an
    inherited stage_a grid (best lr 5e-4), plus fresh seeds on the cd winner."""
    rows = []
    def r(phase, lr, loss, status="completed", **kw):
        rows.append(dict(study="mlp256", optimizer="adamw", phase=phase, lr=lr,
                         wd=0.1, final_val_loss=loss, status=status,
                         cd_sweep="lr" if phase == "cd" else None, **kw))
    for lr, loss in [(1.25e-4, 1.30), (2.5e-4, 1.20), (5e-4, 1.10), (1e-3, 1.05),
                     (2e-3, 1.00), (4e-3, 1.02)]:
        r("cd", lr, loss)
    for lr, loss in [(1.25e-4, 1.25), (2.5e-4, 1.06), (5e-4, 0.90), (1e-3, 0.91),
                     (2e-3, 3.50), (4e-3, None)]:
        r("stage_a", lr, loss, status="completed" if loss else "diverged")
    for s in (1, 2, 3):
        r("fresh_seed", 2e-3, 1.0, seed=s)
    return pd.DataFrame(rows)


def test_lr_sweep_rows_never_mix_tracks():
    df2 = _two_track_df()
    head = an._lr_sweep_rows(df2, "mlp256", "adamw")
    inh = an._lr_sweep_rows(df2, "mlp256", "adamw", track="inherited")
    assert set(head["phase"]) == {"cd"} and len(head) == 6
    assert set(inh["phase"]) == {"stage_a"} and len(inh) == 6
    with pytest.raises(ValueError):
        an._lr_sweep_rows(df2, "mlp256", "adamw", track="both")


def test_basin_and_optima_per_track():
    df2 = _two_track_df()
    # headline cd: best 1.00 at 2e-3; within 1%: 2e-3 only (4e-3 is 1.02)
    assert an.lr_basin_width("mlp256", "adamw", df2, 0.01) == 0.0
    # inherited: best 0.90 at 5e-4; 1e-3 (0.91, +1.1%) is within the 2% tol -> width 1
    assert an.lr_basin_width("mlp256", "adamw", df2, 0.02, track="inherited") == 1.0
    assert an.inherited_best_lr(df2, "mlp256", "adamw") == pytest.approx(5e-4)
    assert an._best_lr(df2, "mlp256", "adamw") == pytest.approx(2e-3)


def test_divergence_rate_uses_headline_track_only():
    df2 = _two_track_df()
    # headline: 6 cd + 3 fresh, none diverged or > best + 0.5 -> 0
    assert an.divergence_rate("mlp256", "adamw", df2) == 0.0
    assert an.headline_tuning_phases(df2, "mlp256") == ("cd",)
    assert an.headline_tuning_phases(df2, "moe") == ("stage_a", "stage_b", "edge")


# ------------------------------------- headline-only diagnostics (H7, fig 8)

def _smoke_moe_logs(root: Path) -> None:
    """MoE smoke runs only (seed 0, one per optimizer): what the real logs
    held before any MoE study ran."""
    for opt in an.OPTIMIZERS:
        cfg = _cfg("moe", opt, "smoke", 0, BEST_LR[opt], 0.01, 1.0)
        h = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
        _write_log(root, cfg, h, MOE_FRESH[opt])


def _h7(out: pd.DataFrame) -> pd.Series:
    return out[out["hypothesis"] == "H7"].iloc[0]


def test_h7_ignores_smoke_logs(df, tmp_path):
    _smoke_moe_logs(tmp_path)
    smoke = an.load_logs(tmp_path)
    assert set(smoke["phase"]) == {"smoke"}
    row = _h7(an.hypothesis_verdicts(df, logs=smoke))
    assert row["verdict"] == "indistinguishable"
    assert row["quantity"].startswith("insufficient data")


def test_h7_uses_fresh_seed_logs(df, logs, tmp_path):
    # world fresh-seed MoE logs give a real verdict; adding smoke logs with
    # extreme entropy must not change it
    base = _h7(an.hypothesis_verdicts(df, logs=logs))
    assert base["verdict"] in {"supported", "falsified"}
    assert not base["quantity"].startswith("insufficient")
    _smoke_moe_logs(tmp_path)
    smoke = an.load_logs(tmp_path)
    smoke.loc[smoke["optimizer"] == "muon", "expert_entropy"] = 99.0
    mixed = pd.concat([logs, smoke], ignore_index=True)
    row = _h7(an.hypothesis_verdicts(df, logs=mixed))
    assert row["verdict"] == base["verdict"]
    assert row["quantity"] == base["quantity"]


def _polluted(logs: pd.DataFrame) -> pd.DataFrame:
    """World logs plus non-headline copies (smoke, cd, 2x budget) of the
    fresh-seed rows carrying extreme values."""
    fresh = logs[logs["phase"] == "fresh_seed"]
    extra = []
    for phase, mult in (("smoke", 1.0), ("cd", 1.0), ("budget_constant", 2.0)):
        e = fresh.copy()
        e["phase"], e["budget_multiplier"], e["seed"] = phase, mult, 0
        ev = e["t"] == "eval"
        e.loc[ev, "max_attn_logit"] = 1e6
        e.loc[ev & e["expert_entropy"].notna(), "expert_entropy"] = -1e6
        fin = e.index[e["t"] == "final"]
        e.loc[fin, "spectral_norms"] = pd.Series(
            [{"layer0.weight": 1e6, "bogus.weight": 1e6}] * len(fin), index=fin)
        extra.append(e)
    return pd.concat([logs] + extra, ignore_index=True)


def test_figure8_data_ignores_non_fresh_rows(logs):
    dirty = _polluted(logs)
    attn, attn_d = an.attn_logit_curves(logs), an.attn_logit_curves(dirty)
    pd.testing.assert_frame_equal(attn, attn_d)
    assert attn["max_attn_logit"].max() < 10
    assert set(attn["study"]) == {"dense", "moe"}
    # mean over the 3 fresh seeds at every point
    assert (attn["n_seeds"] == 3).all()
    ent, ent_d = an.expert_entropy_curves(logs), an.expert_entropy_curves(dirty)
    pd.testing.assert_frame_equal(ent, ent_d)
    assert set(ent["study"]) == {"moe"} and ent["expert_entropy"].min() > 0
    study, sn = an.spectral_norm_means(logs)
    study_d, sn_d = an.spectral_norm_means(dirty)
    assert study == study_d == "dense"
    pd.testing.assert_frame_equal(sn, sn_d)
    assert "bogus.weight" not in sn_d.index


def test_figure8_data_smoke_only_is_empty(tmp_path):
    _smoke_moe_logs(tmp_path)
    smoke = an.load_logs(tmp_path)
    assert an.attn_logit_curves(smoke).empty
    assert an.expert_entropy_curves(smoke).empty
    assert an.spectral_norm_means(smoke)[1].empty
    p = an.figure_8(pd.DataFrame(), tmp_path / "figures", logs=smoke)
    assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_spectral_norm_means_over_seeds_union_layers():
    def row(opt, seed, sn, phase="fresh_seed"):
        return dict(study="dense", optimizer=opt, phase=phase, seed=seed,
                    budget_multiplier=1.0, t="final", spectral_norms=sn)
    q0 = "model.layers.0.self_attn.q_proj.weight"
    rows = [row("muon", 1, {q0: 2.0}), row("muon", 2, {q0: 4.0}),
            row("muon", 0, {q0: 100.0}, phase="smoke"),
            row("muonall", 1, {q0: 3.0, "model.embed_tokens.weight": 5.0}),
            row("adamw", 1, {})]
    study, sn = an.spectral_norm_means(pd.DataFrame(rows))
    assert study == "dense"
    assert list(sn.columns) == ["muon", "muonall"]       # adamw: no norms
    assert list(sn.index) == ["model.embed_tokens.weight", q0]
    assert sn.loc[q0, "muon"] == pytest.approx(3.0)      # mean of seeds 1, 2
    assert np.isnan(sn.loc["model.embed_tokens.weight", "muon"])
    assert sn.loc["model.embed_tokens.weight", "muonall"] == pytest.approx(5.0)


def test_short_layer_name_and_numeric_order():
    assert an.short_layer_name("model.layers.0.self_attn.q_proj.weight") == "L0.q"
    assert an.short_layer_name("model.layers.3.mlp.gate_proj.weight") == "L3.gate"
    assert an.short_layer_name("model.layers.5.self_attn.o_proj.weight") == "L5.o"
    assert an.short_layer_name("model.embed_tokens.weight") == "embed"
    names = [f"model.layers.{i}.{m}.weight"
             for i in (10, 2, 0, 1)
             for m in ("mlp.down_proj", "self_attn.k_proj", "mlp.gate_proj",
                       "self_attn.q_proj", "mlp.up_proj", "self_attn.v_proj",
                       "self_attn.o_proj")]
    names.append("model.embed_tokens.weight")
    short = [an.short_layer_name(n) for n in sorted(names, key=an.layer_sort_key)]
    mods = ["q", "k", "v", "o", "gate", "up", "down"]
    assert short == ["embed"] + [f"L{i}.{m}" for i in (0, 1, 2, 10) for m in mods]
    # MoE experts: compact labels, experts sorted numerically after attention
    assert an.short_layer_name(
        "model.layers.0.mlp.experts.10.gate_proj.weight") == "L0.e10.gate"
    moe = [f"model.layers.{i}.mlp.experts.{e}.{m}_proj.weight"
           for i in (1, 0) for e in (10, 2, 0) for m in ("down", "up", "gate")]
    moe += [f"model.layers.{i}.self_attn.{m}_proj.weight"
            for i in (1, 0) for m in ("o", "q", "v", "k")]
    short = [an.short_layer_name(n) for n in sorted(moe, key=an.layer_sort_key)]
    assert short == [x for i in (0, 1) for x in
                     [f"L{i}.{m}" for m in ("q", "k", "v", "o")]
                     + [f"L{i}.e{e}.{m}" for e in (0, 2, 10)
                        for m in ("gate", "up", "down")]]


def test_partial_and_diagnostics_logs_never_counted(tmp_path):
    """A stopped fresh_seed run (extra seed 4, no completed result) and a
    parked run under logs/diagnostics/ must not be loaded or counted."""
    for seed in (1, 2, 3):
        cfg = _cfg("dense", "adamw", "fresh_seed", seed, BEST_LR["adamw"], 0.01, 1.0)
        h, _ = _write_result(tmp_path, cfg, 3.5)
        _write_log(tmp_path, cfg, h, 3.5)
    # partial seed-4 run: log in logs/dense/, truncated, NO result JSON
    cfg4 = _cfg("dense", "adamw", "fresh_seed", 4, BEST_LR["adamw"], 0.01, 1.0)
    _write_log(tmp_path, cfg4, "partial00004", 3.5)
    p4 = tmp_path / "logs" / "dense" / "partial00004.jsonl"
    lines = p4.read_text().splitlines()
    partial = [json.loads(x) for x in lines[:4]]  # start + 3 evals, stopped
    for rec in partial[1:]:
        rec["max_attn_logit"] = 1e6
    p4.write_text("\n".join(json.dumps(x) for x in partial) + "\n")
    # parked diagnostics runs: a seed-5 fresh_seed copy and a cd audit run
    for name, phase, seed in (("discarded_seeds45/logs/dense", "fresh_seed", 5),
                              ("manifold_audit/scale8", "cd", 0)):
        cfg = _cfg("dense", "adamw", phase, seed, BEST_LR["adamw"], 0.01, 1.0)
        _write_log(tmp_path, cfg, f"diag{seed}", 3.5)
        src = tmp_path / "logs" / "dense" / f"diag{seed}.jsonl"
        dst = tmp_path / "logs" / "diagnostics" / name / f"diag{seed}.jsonl"
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
    # a jsonl nested one level deeper inside a study dir is not read either
    nested = tmp_path / "logs" / "dense" / "old" / "x.jsonl"
    nested.parent.mkdir()
    nested.write_text(dst.read_text())

    lg = an.load_logs(tmp_path)
    res = an.load_results(tmp_path)
    assert set(lg["hash"]) == set(res["hash"]) | {"partial00004"}
    assert not lg["hash"].str.startswith("diag").any()
    assert "cd" not in set(lg["phase"])
    # without df the partial seed 4 is still a fresh_seed log...
    assert an.attn_logit_curves(lg)["n_seeds"].max() == 4
    # ...with df it is dropped: no completed result
    attn = an.attn_logit_curves(lg, res)
    assert (attn["n_seeds"] == 3).all()
    assert attn["max_attn_logit"].max() < 10
    _, sn = an.spectral_norm_means(lg, df=res)
    assert list(sn.columns) == ["adamw"]
