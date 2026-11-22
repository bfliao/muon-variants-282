"""Tests for muonarch/figures.py (docs/figures_spec.md, Engineering).

Two kinds of synthetic input:

* the on-disk synthetic world of tests/test_analysis.py (``make_world``):
  dense tuned Phase-II style, mlp128 tuned by coordinate descent, moe with
  fresh seeds only — used for the "every figure renders" tests and for the
  helpers that need a budget curve or logs;
* small hand-built result frames (``_frame``) where a test needs a precise
  structure: a two-pass coordinate descent, both MLP tuning tracks, stopped
  runs, pilots.

No training code is imported and nothing is written outside tmp dirs.
"""

from __future__ import annotations

import io
import json
import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from muonarch import analysis as an
from muonarch import figures as fg
from tests.test_analysis import (BEST_LR, D1, LION_ACCEPTED, LION_ALPHA_OFF,
                                 LION_BETA2_OFF, LION_OPT_RATE, LION_PLANTED,
                                 LION_WD_OFF, OPT_RATE, PLANTED, STAGE_A_OFF,
                                 WORLD_OPTS, _beyond_budget, curve, make_world)

# ---------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("figworld")
    make_world(root)
    return root


@pytest.fixture(scope="module")
def df(world):
    return an.load_results(world)


@pytest.fixture(scope="module")
def logs(world):
    return an.load_logs(world)


@pytest.fixture(scope="module")
def lion_world(tmp_path_factory):
    """The synthetic world plus Lion-Muon on dense and mlp128."""
    root = tmp_path_factory.mktemp("figlionworld")
    make_world(root, lionmuon=True)
    return root


@pytest.fixture(scope="module")
def lion_df(lion_world):
    return an.load_results(lion_world)


@pytest.fixture(scope="module")
def lion_logs(lion_world):
    return an.load_logs(lion_world)


#: the five-optimizer house order of the base synthetic world
BASE_ORDER = [o for o in fg.OPT_ORDER if o in WORLD_OPTS]

_COUNTER = [0]


def _row(study, opt, phase, seed, lr, loss, *, wd=0.01, momentum=0.95,
         betas=(0.9, 0.95), clip=1.0, status="completed", mult=1.0,
         cd_sweep=None, width=None, fm=100.0, fo=10.0, alpha=30.0, beta2=0.99):
    """One flattened result row.  ``alpha`` / ``beta2`` are Lion-Muon's
    (the ``lionmuon_alpha`` / ``lionmuon_beta2`` columns; None elsewhere)."""
    _COUNTER[0] += 1
    if width is None and study.startswith("mlp"):
        width = int(study.replace("mlp", ""))
    lion = opt == "lionmuon"
    return {
        "study": study, "optimizer": opt, "phase": phase, "seed": seed, "lr": lr,
        "wd": wd, "momentum": None if opt in ("adamw", "lionmuon") else momentum,
        "lionmuon_alpha": alpha if lion else None,
        "lionmuon_beta2": beta2 if lion else None,
        "adamw_betas": list(betas), "grad_clip": clip, "cd_sweep": cd_sweep,
        "budget_multiplier": mult, "status": status,
        "final_val_loss": loss if status == "completed" else None,
        "final_test_acc": 0.7, "width": width, "batch": 256, "seq_len": None,
        "budget_base_steps": 100, "flops_model": fm, "flops_optimizer": fo,
        "flops_total": fm + fo, "hash": f"h{_COUNTER[0]:05d}",
    }


def _frame(rows) -> pd.DataFrame:
    out = pd.DataFrame(rows)
    for c in ("seed", "lr", "wd", "momentum", "final_val_loss", "width",
              "budget_multiplier", "lionmuon_alpha", "lionmuon_beta2"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _two_pass_cd(study="mlp256", opt="muon", inherited=True):
    """A two-pass coordinate descent: the lr sweep at wd=0.01 (pass 1), a wd
    sweep that accepts wd=0.03 (its centre cell keeps the label "wd"), then
    the pass-2 lr sweep at wd=0.03, whose minimum (lr 0.02) differs from the
    pass-1 minimum (lr 0.01).  Fresh seeds use the accepted (0.02, 0.03).
    Optionally an inherited stage_a/stage_b/edge track with its own lrs."""
    rows = []
    for lr, loss in ((0.005, 0.80), (0.01, 0.70), (0.02, 0.74), (0.04, 0.90)):
        rows.append(_row(study, opt, "cd", 0, lr, loss, wd=0.01, cd_sweep="lr"))
    for wd, loss in ((0.0, 0.72), (0.03, 0.65)):
        rows.append(_row(study, opt, "cd", 0, 0.01, loss, wd=wd, cd_sweep="wd"))
    for lr, loss in ((0.005, 0.78), (0.02, 0.60), (0.04, 0.75)):
        rows.append(_row(study, opt, "cd", 0, lr, loss, wd=0.03, cd_sweep="lr"))
    rows.append(_row(study, opt, "cd", 0, 0.08, None, wd=0.03, cd_sweep="lr",
                     status="diverged"))
    for m, loss in ((0.9, 0.63), (0.98, 0.69)):
        rows.append(_row(study, opt, "cd", 0, 0.02, loss, wd=0.03, momentum=m,
                         cd_sweep="momentum"))
    for seed, loss in ((1, 0.61), (2, 0.60), (3, 0.62)):
        rows.append(_row(study, opt, "fresh_seed", seed, 0.02, loss, wd=0.03))
    if inherited:
        for lr, loss in ((0.0007, 0.50), (0.0014, 0.45), (0.0028, 0.55)):
            rows.append(_row(study, opt, "stage_a", 0, lr, loss, wd=0.01))
        rows.append(_row(study, opt, "stage_b", 0, 0.0014, 0.44, wd=0.003))
        rows.append(_row(study, opt, "edge", 0, 0.00035, 0.58, wd=0.01))
    return rows


# ---------------------------------------------------------------- style

def test_style_table_is_complete_and_fixed():
    assert fg.OPT_ORDER == ("adamw", "muon", "moonlight", "muonall", "lionmuon",
                            "manifold")
    assert fg.OPT_LABEL == {"adamw": "AdamW", "muon": "Muon", "moonlight": "Moonlight",
                            "muonall": "MuonAll", "lionmuon": "Lion-Muon",
                            "manifold": "Manifold Muon"}
    assert fg.OPT_COLOR == {"adamw": "#3a3a38", "muon": "#2a78d6",
                            "moonlight": "#eda100", "manifold": "#008300",
                            "muonall": "#e87ba4", "lionmuon": "#6b3fa0"}
    assert fg.OPT_MARKER == {"adamw": "o", "muon": "s", "moonlight": "D",
                             "manifold": "^", "muonall": "v", "lionmuon": "P"}
    # every optimizer the analysis knows is styled, and nothing else
    for table in (fg.OPT_LABEL, fg.OPT_COLOR, fg.OPT_MARKER):
        assert set(table) == set(an.OPTIMIZERS) == set(fg.OPT_ORDER)
    assert len(fg.OPT_ORDER) == len(set(fg.OPT_ORDER)) == 6
    assert len(set(fg.OPT_COLOR.values())) == 6
    assert len(set(fg.OPT_MARKER.values())) == 6
    assert len(set(fg.OPT_LABEL.values())) == 6
    # the legacy analysis figures style every optimizer too
    assert set(an.OPTIMIZER_COLORS) == set(an.OPTIMIZER_MARKERS) == set(an.OPTIMIZERS)
    assert len(set(an.OPTIMIZER_COLORS.values())) == 6
    # the five original optimizers keep their relative order
    assert BASE_ORDER == ["adamw", "muon", "moonlight", "muonall", "manifold"]
    # F6 columns follow the coordinate-descent order; alpha is Lion-Muon's
    assert fg.CD_HYPERS == ("lr", "wd", "alpha", "momentum", "beta2", "clip")
    assert set(fg.CD_HYPER_LABEL) == set(fg.CD_HYPER_LABEL_NARROW) == set(fg.CD_HYPERS)
    assert fg.CD_HYPER_OWNER == {"alpha": "lionmuon"}
    assert fg.STUDY_LABEL["mlp128"] == "MLP-128"
    assert fg.STUDY_LABEL["mlp256"] == "MLP-256"
    assert fg.STUDY_LABEL["mlp512"] == "MLP-512"
    assert fg.STUDY_LABEL["dense"] == "Dense transformer"
    assert fg.WIDTH_COLOR == {128: "#86b6ef", 256: "#2a78d6", 512: "#0d366b"}
    assert (fg.INK, fg.INK2, fg.GRID) == ("#0b0b0b", "#52514e", "#e6e5e0")
    assert (fg.FULL_W, fg.COL_W, fg.DPI) == (7.2, 3.5, 200)


def test_style_context_sets_house_rcparams():
    before = plt.rcParams["font.size"]
    with fg.style():
        rc = plt.rcParams
        assert rc["font.size"] == 9 and rc["axes.titlesize"] == 10
        assert rc["xtick.labelsize"] == 8 and rc["ytick.labelsize"] == 8
        assert rc["axes.spines.top"] is False and rc["axes.spines.right"] is False
        assert rc["grid.linestyle"] == "-" and rc["grid.color"] == fg.GRID
        assert rc["axes.axisbelow"] is True
        assert rc["figure.constrained_layout.use"] is True
        assert rc["text.color"] == fg.INK and rc["axes.labelcolor"] == fg.INK
        assert rc["savefig.dpi"] == 200
    assert plt.rcParams["font.size"] == before          # restored on exit


def test_formatters():
    assert fg.fmt_mult(2.31) == "2.3×"
    assert fg.fmt_mult(0.999) == "1.0×"
    assert fg.fmt_mult(0.642) == "0.64×"
    assert fg.fmt_mult(0.8) == "0.80×"
    assert fg.fmt_mult(0.0391) == "0.039×"
    assert fg.fmt_mult(21.2) == "21×"
    assert fg.fmt_mult(math.inf) == "∞"
    assert fg.fmt_shift(0.5) == "÷2"
    assert fg.fmt_shift(4.0) == "×4"
    assert fg.fmt_shift(1.0) == "no shift"
    assert fg.fmt_num(0.003125) == "0.003125"
    assert fg.fmt_num(math.inf) == "none"
    assert fg.fmt_cat(0.003) == ".003" and fg.fmt_cat(1.0) == "1"
    assert fg.fmt_cat(0.0) == "0" and fg.fmt_cat(math.inf) == "none"


def test_nice_log_ticks():
    assert fg.nice_log_ticks(0.008, 5.0) == [0.01, 0.1, 1.0]
    t = fg.nice_log_ticks(0.7, 3.9, max_n=5)
    assert t == pytest.approx([0.7, 1.0, 1.5, 2.0, 3.0])
    assert fg.nice_log_ticks(1.0, 1.0) == []
    for lo, hi in ((1.2e14, 3.6e14), (3e11, 6e13), (1e12, 2e15)):
        t = fg.nice_log_ticks(lo, hi, max_n=3)
        assert 1 <= len(t) <= 3 and all(lo <= v <= hi for v in t)


# ---------------------------------------------------------------- headline runs

def test_headline_studies_fixed_order_and_no_moe(df):
    # the world has dense, mlp128 and moe fresh seeds: moe is not a headline
    # study, and the order is the fixed display order
    assert fg.headline_studies(df) == ["mlp128", "dense"]
    assert fg.headline_studies(df.iloc[0:0]) == []


def test_headline_runs_only_fresh_seeds(df):
    runs = fg.headline_runs(df, "mlp128")
    assert len(runs) == 15
    assert list(runs["optimizer"].unique()) == BASE_ORDER
    assert set(runs["seed"]) == {1, 2, 3}
    fresh = an._fresh(df, "mlp128")
    assert sorted(runs["loss"]) == pytest.approx(sorted(fresh["final_val_loss"]))


def test_headline_runs_keep_diverged_drop_parked_and_partial():
    rows = _two_pass_cd()
    rows.append(_row("mlp256", "muon", "fresh_seed", 4, 0.02, None, wd=0.03,
                     status="diverged"))
    rows.append(_row("mlp256", "muon", "fresh_seed", 5, 0.02, 0.10, wd=0.03,
                     status="stopped"))               # parked / partial run
    rows.append(_row("mlp256", "muon", "fresh_seed", 1, 0.02, 0.20, wd=0.03,
                     mult=2.0))                       # not the 1x budget
    d = _frame(rows)
    runs = fg.headline_runs(d, "mlp256")
    assert list(runs["status"]).count("completed") == 3
    assert list(runs["status"]).count("diverged") == 1     # drawn, not dropped
    assert runs["loss"].isna().sum() == 1
    assert 0.10 not in set(runs["loss"].dropna())          # stopped run
    assert 0.20 not in set(runs["loss"].dropna())          # 2x run
    assert runs["loss"].min() >= 0.60                      # no tuning rows
    # rank / cost helpers count completed fresh seeds only
    assert fg.rank_table(d)["mean_loss"].iloc[0] == pytest.approx(0.61)


def test_parked_logs_never_enter_curves(world, df, logs):
    cur = fg.loss_curves(df, logs, "mlp128", "tokens")
    assert set(cur["n_seeds"]) == {3}
    # a stopped fresh_seed run: logs on disk, no completed result
    victim = an._fresh(df, "mlp128", "muon").iloc[0]["hash"]
    d2 = df.copy()
    d2.loc[d2["hash"] == victim, "status"] = "stopped"
    cur2 = fg.loss_curves(d2, logs, "mlp128", "tokens")
    assert set(cur2[cur2["optimizer"] == "muon"]["n_seeds"]) == {2}
    assert set(cur2[cur2["optimizer"] == "adamw"]["n_seeds"]) == {3}
    assert len(fg.headline_runs(d2, "mlp128")) == 14


# ---------------------------------------------------------------- CD helpers

def test_accepted_config_is_the_fresh_seed_config():
    d = _frame(_two_pass_cd())
    acc = fg.accepted_config(d, "mlp256", "muon")
    assert acc == {"lr": 0.02, "wd": 0.03, "alpha": None, "momentum": 0.95,
                   "beta2": None, "clip": 1.0}
    assert fg.accepted_config(d, "mlp256", "adamw") is None
    # AdamW: momentum means beta1, beta2 exists, no clipping -> +inf
    a = _frame([_row("mlp128", "adamw", "fresh_seed", 1, 0.001, 0.9,
                     betas=(0.8, 0.99), clip=None)])
    assert fg.accepted_config(a, "mlp128", "adamw") == {
        "lr": 0.001, "wd": 0.01, "alpha": None, "momentum": 0.8, "beta2": 0.99,
        "clip": math.inf}
    # Lion-Muon: no momentum; beta2 is ITS EMA decay (not adamw_betas[1],
    # which is the fixed (0.9, 0.95) of its AdamW group); alpha exists
    lm = _frame([_row("mlp128", "lionmuon", "fresh_seed", 1, 0.001, 0.9,
                      alpha=100.0, beta2=0.9, betas=(0.9, 0.95))])
    assert fg.accepted_config(lm, "mlp128", "lionmuon") == {
        "lr": 0.001, "wd": 0.01, "alpha": 100.0, "momentum": None, "beta2": 0.9,
        "clip": 1.0}


def test_cd_slices_structure():
    d = _frame(_two_pass_cd())
    sl = fg.cd_slices(d, "mlp256", "muon")
    best = 0.60
    # lr: the final slice sits at the accepted wd=0.03 and holds the centre
    # cell even though dedup left it labelled "wd"; the pass-1 slice is kept
    # as an earlier pass
    lr = sl[sl["hyper"] == "lr"]
    fin = lr[lr["final"]]
    assert sorted(fin["value"]) == [0.005, 0.01, 0.02, 0.04, 0.08]
    assert fin[fin["value"] == 0.01]["loss"].iloc[0] == pytest.approx(0.65)
    assert fin[fin["value"] == 0.08]["status"].iloc[0] == "diverged"
    assert list(fin[fin["accepted"]]["value"]) == [0.02]
    early = lr[~lr["final"]]
    assert sorted(early["value"]) == [0.005, 0.01, 0.02, 0.04]
    assert not early["accepted"].any()
    assert early["slice"].nunique() == 1 and fin["slice"].nunique() == 1
    assert set(early["slice"]) != set(fin["slice"])
    # delta is relative to the optimizer's best completed cd loss, >= 0
    ok = sl[sl["status"] == "completed"]
    assert (ok["delta"] >= -1e-12).all()
    assert ok["delta"].min() == pytest.approx(0.0)
    assert fin[fin["value"] == 0.02]["delta"].iloc[0] == pytest.approx(0.60 - best)
    # wd: ONE real sweep (lr 0.01); two lr sweeps at different wd do not
    # masquerade as two-point wd sweeps at lr 0.005 / 0.02 / 0.04
    wd = sl[sl["hyper"] == "wd"]
    assert wd["slice"].nunique() == 1
    assert sorted(wd["value"]) == [0.0, 0.01, 0.03]
    assert not wd["final"].any()        # swept at the pre-acceptance lr
    # momentum: final slice with the accepted centre
    mo = sl[sl["hyper"] == "momentum"]
    assert sorted(mo["value"]) == [0.9, 0.95, 0.98] and mo["final"].all()
    assert list(mo[mo["accepted"]]["value"]) == [0.95]
    # hyperparameters the optimizer does not have / did not sweep: no rows
    assert sl[sl["hyper"].isin(["beta2", "clip"])].empty
    assert fg.cd_slices(d, "mlp256", "adamw").empty


def test_cd_slices_unlabelled_frames_skip_the_label_requirement():
    rows = _two_pass_cd()
    for r in rows:
        r["cd_sweep"] = None
    sl = fg.cd_slices(_frame(rows), "mlp256", "muon")
    fin = sl[(sl["hyper"] == "lr") & sl["final"]]
    assert sorted(fin["value"]) == [0.005, 0.01, 0.02, 0.04, 0.08]


def test_lr_sweep_is_the_accepted_context_slice():
    d = _frame(_two_pass_cd())
    sw = fg.lr_sweep(d, "mlp256", "muon")
    assert list(sw["lr"]) == [0.005, 0.01, 0.02, 0.04, 0.08]
    ok = sw[sw["status"] == "completed"]
    assert float(ok.loc[ok["loss"].idxmin(), "lr"]) == 0.02      # pass-2 minimum
    assert list(sw[sw["status"] == "diverged"]["lr"]) == [0.08]
    assert an._best_lr(d, "mlp256", "muon") == 0.02


def test_lr_sweep_falls_back_to_stage_a_without_cd(df):
    sw = fg.lr_sweep(df, "dense", "muonall")
    rows = df[(df["study"] == "dense") & (df["optimizer"] == "muonall")
              & (df["phase"] == "stage_a")]
    assert sorted(sw["lr"]) == pytest.approx(sorted(rows["lr"]))
    assert (sw["status"] == "diverged").sum() == 1                # kept, not dropped


def test_inherited_track_never_enters_headline_helpers():
    d = _frame(_two_pass_cd(inherited=True))
    inherited_lrs = {0.0007, 0.0014, 0.0028, 0.00035}
    inherited_losses = {0.50, 0.45, 0.55, 0.44, 0.58}
    assert an.headline_tuning_phases(d, "mlp256") == ("cd",)
    sw = fg.lr_sweep(d, "mlp256", "muon")
    assert not set(sw["lr"]) & inherited_lrs
    sl = fg.cd_slices(d, "mlp256", "muon")
    assert not set(sl[sl["hyper"] == "lr"]["value"]) & inherited_lrs
    assert not set(sl["loss"].dropna()) & inherited_losses
    st = fg.lr_stability(d, "mlp256", "muon")
    assert not set(st["lr"]) & inherited_lrs
    runs = fg.headline_runs(d, "mlp256")
    assert not set(runs["loss"].dropna()) & inherited_losses
    # ... and identical results with the inherited rows removed
    d0 = _frame(_two_pass_cd(inherited=False))
    pd.testing.assert_frame_equal(sw, fg.lr_sweep(d0, "mlp256", "muon"))
    pd.testing.assert_frame_equal(st, fg.lr_stability(d0, "mlp256", "muon"))
    # the inherited optimum is reported only by the dedicated helpers
    tab = fg.optimal_lr_by_width(d)
    assert tab[tab["track"] == "inherited"]["lr"].iloc[0] == 0.0014
    assert tab[tab["track"] == "retuned"]["lr"].iloc[0] == 0.02


def test_noise_threshold():
    rows = [_row("mlp128", "adamw", "pilot", s, 0.001, l)
            for s, l in ((101, 0.80), (102, 0.82), (103, 0.84))]
    d = _frame(rows)
    assert fg.noise_threshold(d, "mlp128") == pytest.approx(2 * 0.02)
    assert math.isnan(fg.noise_threshold(d, "mlp256"))             # no pilot
    assert math.isnan(fg.noise_threshold(_frame(rows[:1]), "mlp128"))
    rows.append(_row("mlp128", "adamw", "pilot", 104, 0.001, None, status="diverged"))
    assert math.isinf(fg.noise_threshold(_frame(rows), "mlp128"))


def test_lr_stability_states():
    rows = _two_pass_cd(inherited=False)
    rows.append(_row("mlp256", "muon", "cd", 0, 0.0001, 1.50, wd=0.03, cd_sweep="lr"))
    st = fg.lr_stability(_frame(rows), "mlp256", "muon").set_index("lr")
    assert st.loc[0.08, "state"] == "diverged"
    assert st.loc[0.0001, "state"] == "high_loss"       # > best + 0.5 nats
    assert st.loc[0.02, "state"] == "ok" and bool(st.loc[0.02, "best"])
    assert int(st["best"].sum()) == 1


# ---------------------------------------------------------------- speedups, curves

def test_speedup_summary_recovers_planted(df):
    tok = fg.speedup_summary(df, "dense", "tokens").set_index("optimizer")
    for opt, s in PLANTED["dense"].items():
        r = tok.loc[opt]
        assert r["lo"] <= r["mid"] <= r["hi"]
        assert r["mid"] == pytest.approx(s, rel=0.02)
        assert r["n_inf"] == 0
    assert list(tok.index) == [o for o in fg.OPT_ORDER if o in tok.index]
    fl = fg.speedup_summary(df, "dense", "flops").set_index("optimizer")
    assert fl.loc["manifold", "hi"] < 0.5
    assert fg.speedup_summary(df, "moe", "tokens").empty   # no budget curve


def test_budget_fits_in_budget_multiples(df):
    fits = fg.budget_fits(df, "mlp128")
    assert sorted(fits) == [1, 2, 3]
    pts, fit, d1 = fits[1]
    assert d1 == D1["mlp128"]
    assert [p[0] for p in pts] == pytest.approx([0.125, 0.25, 0.5, 1.0, 2.0])
    for x, loss in pts:
        assert fit(x * d1) == pytest.approx(loss, abs=1e-3)
    assert fg.budget_fits(df, "moe") == {}


def test_loss_curves_tokens_and_flops(df, logs):
    tok = fg.loss_curves(df, logs, "dense", "tokens")
    assert set(tok["optimizer"]) == set(WORLD_OPTS)
    g = tok[tok["optimizer"] == "muon"].sort_values("k")
    assert g["x"].iloc[-1] == pytest.approx(D1["dense"])
    assert (g["lo"] <= g["mean"] + 1e-12).all() and (g["mean"] <= g["hi"] + 1e-12).all()
    final = an._fresh(df, "dense", "muon")["final_val_loss"].mean()
    assert g["mean"].iloc[-1] == pytest.approx(final, abs=0.05)
    # FLOP axis: the run's measured total FLOPs x the fraction of budget done
    fl = fg.loss_curves(df, logs, "dense", "flops")
    gf = fl[fl["optimizer"] == "muon"].sort_values("k")
    total = an._fresh(df, "dense", "muon")["flops_total"].mean()
    assert gf["x"].iloc[-1] == pytest.approx(total)
    assert gf["mean"].to_numpy() == pytest.approx(g["mean"].to_numpy())
    ratio = (fl[fl["optimizer"] == "manifold"]["x"].max()
             / fl[fl["optimizer"] == "adamw"]["x"].max())
    assert ratio == pytest.approx((1 + OPT_RATE["manifold"]) / (1 + OPT_RATE["adamw"]))
    assert fg.loss_curves(df, None, "dense").empty


def test_loss_curves_append_final_when_last_eval_is_short(df, logs):
    # drop every run's last eval: the final validation loss is appended at
    # the end of the budget instead
    ev = logs[(logs["t"] == "eval") & (logs["study"] == "mlp128")]
    last = ev.groupby("hash")["images_or_tokens"].transform("max")
    cut = logs.drop(ev[ev["images_or_tokens"] == last].index)
    cur = fg.loss_curves(df, cut, "mlp128", "tokens")
    g = cur[cur["optimizer"] == "adamw"].sort_values("k")
    assert g["x"].iloc[-1] == pytest.approx(D1["mlp128"])
    assert g["mean"].iloc[-1] == pytest.approx(
        an._fresh(df, "mlp128", "adamw")["final_val_loss"].mean())


def test_compute_cost(df):
    c = fg.compute_cost(df, "dense").set_index("optimizer")
    for opt, rate in OPT_RATE.items():
        assert c.loc[opt, "model"] == 1.0
        assert c.loc[opt, "optimizer_cost"] == pytest.approx(rate)
        assert c.loc[opt, "total"] == pytest.approx(1 + rate)


def test_rank_table(df):
    tab = fg.rank_table(df)
    assert set(tab["study"]) == {"mlp128", "dense"}
    d = tab[tab["study"] == "dense"].set_index("optimizer")
    # planted dense speedups: muon 1.3 > moonlight 1.15 > muonall 1.0 = adamw
    # > manifold 0.9
    assert d.loc["muon", "rank"] == 1 and d.loc["moonlight", "rank"] == 2
    assert d.loc["manifold", "rank"] == 5
    assert sorted(d["rank"]) == [1, 2, 3, 4, 5]
    assert bool(d.loc["muon", "separated"]) is True       # rank 1 by definition
    assert bool(d.loc["moonlight", "separated"]) is True  # clear gap, tiny seed std
    # muonall and adamw are planted identical: whichever is ranked lower is
    # NOT separated from the one above
    lower = d.loc[["muonall", "adamw"]].sort_values("rank").iloc[1]
    assert bool(lower["separated"]) is False


# ---------------------------------------------------------------- width / tracks

def _width_frame():
    rows = []
    best = {"mlp128": 0.001, "mlp256": 0.001, "mlp512": 0.0005}
    for study, lr in best.items():
        for opt in ("adamw", "muon"):
            olr = lr if opt == "adamw" else 0.005
            for seed in (1, 2, 3):
                rows.append(_row(study, opt, "fresh_seed", seed, olr, 0.8 + 0.01 * seed))
            for k, off in ((0.5, 0.03), (1, 0.0), (2, 0.02)):
                rows.append(_row(study, opt, "cd", 0, olr * k, 0.8 + off, cd_sweep="lr"))
            if study != "mlp128":
                for k, off in ((1, 0.01), (2, 0.0), (4, 0.03)):
                    rows.append(_row(study, opt, "stage_a", 0, olr * k, 0.79 + off))
    return _frame(rows)


def test_optimal_lr_by_width():
    tab = fg.optimal_lr_by_width(_width_frame())
    ret = tab[(tab["track"] == "retuned") & (tab["optimizer"] == "adamw")]
    assert list(ret.sort_values("width")["width"]) == [128, 256, 512]
    assert list(ret.sort_values("width")["lr"]) == [0.001, 0.001, 0.0005]
    inh = tab[(tab["track"] == "inherited") & (tab["optimizer"] == "adamw")]
    assert list(inh.sort_values("width")["width"]) == [256, 512]    # none at 128
    assert list(inh.sort_values("width")["lr"]) == [0.002, 0.001]


def test_f5b_markers_sit_exactly_on_the_widths():
    d = _width_frame()
    fig = fg.build_f5b(d)
    try:
        tab = fg.optimal_lr_by_width(d)
        seen = 0
        for ax in fig.axes:
            for line in ax.get_lines():
                if line.get_gid() not in ("retuned", "inherited"):
                    continue
                seen += 1
                xs = [float(v) for v in line.get_xdata()]
                assert set(xs) <= {128.0, 256.0, 512.0}, xs          # no dodging
                opt = {v: k for k, v in fg.OPT_LABEL.items()}[ax.get_title()]
                t = tab[(tab["optimizer"] == opt) & (tab["track"] == line.get_gid())]
                assert xs == [float(w) for w in t.sort_values("width")["width"]]
                assert list(line.get_ydata()) == pytest.approx(
                    list(t.sort_values("width")["lr"]))
                # colour and marker follow the optimizer
                assert line.get_color() == fg.OPT_COLOR[opt]
                assert line.get_marker() == fg.OPT_MARKER[opt]
            assert [float(v) for v in ax.get_xticks()] == [128.0, 256.0, 512.0]
        assert seen == 4            # 2 optimizers x 2 tracks
    finally:
        plt.close(fig)


def test_track_comparison():
    tab = fg.track_comparison(_frame(_two_pass_cd(inherited=True)))
    assert len(tab) == 1
    r = tab.iloc[0]
    assert (r["study"], r["optimizer"]) == ("mlp256", "muon")
    assert r["retuned_loss"] == pytest.approx(0.60) and r["retuned_lr"] == 0.02
    assert r["inherited_loss"] == pytest.approx(0.44) and r["inherited_lr"] == 0.0014
    assert fg.track_comparison(_frame(_two_pass_cd(inherited=False))).empty


# ---------------------------------------------------------------- robustness

def test_robustness_helpers():
    rows = _two_pass_cd(inherited=False)
    rows.append(_row("mlp256", "muon", "faithful", 1, 0.02, 0.615, wd=0.03))
    for seed, loss in ((1, 0.64), (2, 0.65)):
        rows.append(_row("mlp256", "muon", "init_ablation", seed, 0.02, loss, wd=0.03))
    for seed, loss in ((1, 0.50), (2, 0.51)):
        rows.append(_row("mlp256", "adamw", "budget_constant", seed, 0.001, loss,
                         mult=2.0))
    rows.append(_row("mlp256", "adamw", "budget_halflr", 1, 0.0005, 0.53, mult=2.0))
    d = _frame(rows)
    fa = fg.faithful_pairs(d)
    assert len(fa) == 1
    assert fa.iloc[0]["standard"] == pytest.approx(0.61)        # the same seed
    assert fa.iloc[0]["faithful"] == pytest.approx(0.615)
    assert (fa.iloc[0]["lo"], fa.iloc[0]["hi"]) == (0.60, 0.62)
    ia = fg.init_ablation_runs(d)
    assert sorted(ia[ia["variant"] == "stiefel_init"]["loss"]) == [0.64, 0.65]
    assert sorted(ia[ia["variant"] == "standard"]["loss"]) == [0.60, 0.61, 0.62]
    hl = fg.halflr_pairs(d)
    assert len(hl) == 1 and hl.iloc[0]["seed"] == 1
    assert (hl.iloc[0]["full_lr"], hl.iloc[0]["half_lr"]) == (0.50, 0.53)


def test_manifold_audit_rows(tmp_path):
    assert fg.manifold_audit_rows(tmp_path).empty
    d = tmp_path / "logs" / "diagnostics" / "manifold_audit"
    d.mkdir(parents=True)
    (d / "summary.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"name": "baseline_replica", "val": 0.9156, "status": "completed"},
        {"name": "scale8", "val": None, "status": "diverged"},
        {"name": "muon_seed0_reference", "val": 0.7981, "status": "completed"},
        {"name": "something_new", "val": 0.9, "status": "completed"},
    ]) + "\n")
    aud = fg.manifold_audit_rows(tmp_path).set_index("name")
    assert aud.loc["baseline_replica", "loss"] == pytest.approx(0.9156)
    assert aud.loc["baseline_replica", "optimizer"] == "manifold"
    assert aud.loc["scale8", "status"] == "diverged"
    assert math.isnan(aud.loc["scale8", "loss"])
    assert aud.loc["muon_seed0_reference", "optimizer"] == "muon"
    assert aud.loc["something_new", "label"] == "something_new"   # unknown: raw name


def test_spectral_and_attention_helpers(df, logs):
    sp = fg.spectral_by_layer(logs, df, "dense")
    assert set(sp["optimizer"]) <= set(an.OPTIMIZERS)
    assert set(sp["short"]) == {"layer0", "layer1", "layer2"}
    assert (sp["sigma"] > 0).all()
    at = fg.attn_logit_bands(logs, df, "dense")
    assert set(at["n_seeds"]) == {3}
    assert (at["lo"] <= at["hi"]).all()
    assert fg.attn_logit_bands(logs, df, "mlp128").empty
    real = "model.layers.3.mlp.gate_proj.weight"
    fake_logs = logs.copy()
    fake_logs["spectral_norms"] = fake_logs["spectral_norms"].apply(
        lambda v: {real: 2.0} if isinstance(v, dict) else v)
    sp2 = fg.spectral_by_layer(fake_logs, df, "dense")
    assert set(sp2["layer"]) == {3} and set(sp2["kind"]) == {"gate"}


# ---------------------------------------------------------------- rendering

def _fig_ids(d):
    return [fid for fid, _ in fg.figure_plan(d)]


def test_figure_plan_lists_every_figure(df):
    ids = _fig_ids(df)
    for fid in ("f1_final_loss", "f2_speedup", "f3_budget_curve",
                "f4_loss_curves_tokens", "f4b_loss_curves_flops", "f5_width_transfer",
                "f5b_optimal_lr_vs_width", "f6_hparam_sensitivity_mlp128",
                "f7_compute_cost", "f8_rank_summary", "a1_lr_sensitivity",
                "a2_inherited_vs_retuned", "a3_robustness", "a4_loss_vs_accuracy",
                "a5_manifold_audit", "a6_attention_logits",
                "a7_spectral_norms_by_layer", "a8_divergence", "moe_figures"):
        assert fid in ids
    # F6 exists only for studies with coordinate-descent data
    assert "f6_hparam_sensitivity_dense" not in ids
    assert len(ids) == len(set(ids))


def test_every_figure_renders_on_the_synthetic_world(df, logs, tmp_path):
    out = tmp_path / "figs"
    buf = io.StringIO()
    res = fg.build_all(df, logs, out, out=buf)
    lines = buf.getvalue().strip().splitlines()
    assert len(lines) == len(res) == len(fg.figure_plan(df))      # one line each
    written = {k: v for k, v in res.items() if v}
    for fid, paths in written.items():
        assert [p.name for p in paths] == [f"{fid}.png", f"{fid}.pdf"]
        for p in paths:
            assert p.parent == out and p.exists() and p.stat().st_size > 1000
    for fid in ("f1_final_loss", "f2_speedup", "f3_budget_curve",
                "f4_loss_curves_tokens", "f4b_loss_curves_flops", "f5_width_transfer",
                "f5b_optimal_lr_vs_width", "f6_hparam_sensitivity_mlp128",
                "f7_compute_cost", "f8_rank_summary", "a1_lr_sensitivity",
                "a3_robustness", "a4_loss_vs_accuracy", "a5_manifold_audit",
                "a6_attention_logits", "a8_divergence"):
        assert fid in written, fid
    # skipped figures say why, on their own line
    for fid in set(res) - set(written):
        line = next(l for l in lines if l.startswith(fid + ":"))
        assert "skipped" in line and len(line.split("—", 1)[1].strip()) > 5
    for fid in written:
        line = next(l for l in lines if l.startswith(fid + ":"))
        assert "written" in line
    # nothing but the figures was written, and no figure was left open
    assert sorted(p.name for p in out.iterdir()) == sorted(
        p.name for v in written.values() for p in v)
    assert plt.get_fignums() == []
    # 200 dpi PNG at the two-column width
    from matplotlib import image
    w = image.imread(out / "f1_final_loss.png").shape[1]
    assert w == round(fg.FULL_W * fg.DPI)


def test_each_fig_function_returns_its_paths(df, logs, tmp_path):
    for fid, fn in fg.figure_plan(df):
        paths = fn(df, logs, tmp_path)
        assert isinstance(paths, list)
        assert all(p.exists() for p in paths)
        if not paths:
            assert fid in fg.SKIP_REASONS
    plt.close("all")


def test_figures_render_on_hand_built_frames(tmp_path):
    # both MLP tuning tracks, a two-pass CD with a diverged cell, pilots,
    # robustness runs — and no logs at all
    rows = _two_pass_cd(inherited=True)
    rows += [_row("mlp256", "adamw", "pilot", s, 0.001, l)
             for s, l in ((101, 0.80), (102, 0.81), (103, 0.82))]
    rows.append(_row("mlp256", "muon", "faithful", 1, 0.02, 0.615, wd=0.03))
    rows.append(_row("mlp256", "muon", "fresh_seed", 4, 0.02, None, wd=0.03,
                     status="diverged"))
    d = _frame(rows)
    buf = io.StringIO()
    res = fg.build_all(d, None, tmp_path, out=buf)
    for fid in ("f1_final_loss", "f5_width_transfer", "f5b_optimal_lr_vs_width",
                "f6_hparam_sensitivity_mlp256", "f7_compute_cost", "f8_rank_summary",
                "a1_lr_sensitivity", "a2_inherited_vs_retuned", "a3_robustness",
                "a8_divergence"):
        assert res[fid], (fid, buf.getvalue())
    # no budget curve, no logs: those figures are skipped cleanly
    for fid in ("f2_speedup", "f3_budget_curve", "f4_loss_curves_tokens",
                "f4b_loss_curves_flops", "a6_attention_logits"):
        assert res[fid] == []
    assert plt.get_fignums() == []


def test_empty_results_skip_everything(tmp_path):
    buf = io.StringIO()
    res = fg.build_all(pd.DataFrame(), None, tmp_path / "none", out=buf)
    assert all(v == [] for v in res.values())
    assert all("skipped" in l for l in buf.getvalue().strip().splitlines())
    assert not (tmp_path / "none").exists()


def test_moe_figures_are_skipped_cleanly(df, logs, tmp_path):
    # the real project state: MoE has smoke runs only
    no_moe = df[~((df["study"] == "moe") & (df["phase"] == "fresh_seed"))]
    assert not fg.moe_ready(no_moe)
    assert fg.fig_moe(no_moe, logs, tmp_path) == []
    assert "no MoE fresh-seed runs" in fg.SKIP_REASONS["moe_figures"]
    # MoE never appears in a headline figure even once its seeds exist
    assert fg.moe_ready(df)
    assert "moe" not in fg.headline_studies(df)
    assert set(fg.rank_table(df)["study"]) == {"mlp128", "dense"}
    assert list(tmp_path.iterdir()) == []


def test_main_writes_into_the_given_outdir(world, tmp_path, capsys):
    out = tmp_path / "out"
    assert fg.main([str(world), str(out)]) == 0
    printed = capsys.readouterr().out.strip().splitlines()
    assert len(printed) == len(fg.figure_plan(an.load_results(world)))
    assert (out / "f1_final_loss.png").exists() and (out / "f1_final_loss.pdf").exists()
    # the synthetic world itself is untouched
    assert not (world / "figures").exists()
    assert np.all([l.split(":")[0] in _fig_ids(an.load_results(world)) for l in printed])


# ---------------------------------------------------------------- review fixes

def test_lr_sweep_is_a_view_of_the_analysis_sweep():
    # ONE implementation: figures.lr_sweep == analysis._lr_sweep_rows
    d = _frame(_two_pass_cd())
    rows = an._lr_sweep_rows(d, "mlp256", "muon")
    sw = fg.lr_sweep(d, "mlp256", "muon")
    assert sorted(sw["lr"]) == sorted(rows["lr"])
    ok = rows[rows["status"] == "completed"]
    assert sorted(sw["loss"].dropna()) == pytest.approx(sorted(ok["final_val_loss"]))
    assert set(np.round(rows["wd"], 12)) == {0.03}             # accepted context


def test_lr_sweep_when_the_accepted_centre_beats_every_lr_labelled_row():
    # the centre (lr 0.01, wd 0.03) carries the label "wd" and is the best
    # cell; the best lr-LABELLED cell is in pass 1 (wd 0.01)
    rows = []
    for lr, loss in ((0.005, 0.80), (0.01, 0.700), (0.02, 0.705), (0.04, 0.90)):
        rows.append(_row("mlp512", "moonlight", "cd", 0, lr, loss, wd=0.01,
                         cd_sweep="lr"))
    rows.append(_row("mlp512", "moonlight", "cd", 0, 0.01, 0.65, wd=0.03,
                     cd_sweep="wd"))
    for lr, loss in ((0.005, 0.78), (0.02, 0.72), (0.04, 0.85)):
        rows.append(_row("mlp512", "moonlight", "cd", 0, lr, loss, wd=0.03,
                         cd_sweep="lr"))
    rows.append(_row("mlp512", "moonlight", "cd", 0, 0.08, None, wd=0.01,
                     cd_sweep="lr", status="diverged"))        # pass-1 divergence
    for seed in (1, 2, 3):
        rows.append(_row("mlp512", "moonlight", "fresh_seed", seed, 0.01, 0.65,
                         wd=0.03))
    d = _frame(rows)
    sw = fg.lr_sweep(d, "mlp512", "moonlight")
    assert list(sw["lr"]) == [0.005, 0.01, 0.02, 0.04]
    assert list(sw["loss"]) == pytest.approx([0.78, 0.65, 0.72, 0.85])
    assert an.lr_basin_width("mlp512", "moonlight", d, 0.01) == 0.0
    # the pass-1 divergence is not on the curve but is reported for the x mark
    assert fg.earlier_pass_divergences(d, "mlp512", "moonlight") == [0.08]


def test_earlier_pass_divergences_exclude_the_shown_sweep():
    rows = _two_pass_cd(inherited=False)        # diverged lr 0.08 IN the final sweep
    d = _frame(rows)
    assert fg.earlier_pass_divergences(d, "mlp256", "muon") == []
    rows.append(_row("mlp256", "muon", "cd", 0, 0.16, None, wd=0.01, cd_sweep="lr",
                     status="diverged"))
    assert fg.earlier_pass_divergences(_frame(rows), "mlp256", "muon") == [0.16]
    assert fg.earlier_pass_divergences(_frame(rows), "mlp256", "adamw") == []


def test_speedup_centre_is_the_geometric_mean(df, monkeypatch):
    mat = [[1.0, 2.0, 4.0], [1.0, 2.0, 4.0], [1.0, 2.0, 16.0]]
    fake = pd.DataFrame({"speedup_min": [1.0], "speedup_max": [16.0],
                         "matrix": [mat]}, index=["muon"])
    monkeypatch.setattr(an, "token_speedup_range", lambda study, d: fake)
    r = fg.speedup_summary(df, "dense", "tokens").set_index("optimizer").loc["muon"]
    flat = np.array(mat).ravel()
    geo = float(np.exp(np.log(flat).mean()))
    assert r["mid"] == pytest.approx(geo)
    assert r["mid"] != pytest.approx(flat.mean(), rel=0.05)     # not the arithmetic mean
    assert r["mid"] != pytest.approx(float(np.median(flat)), rel=0.05)
    assert (r["lo"], r["hi"]) == (1.0, 16.0)
    assert (r["n_curve"], r["n_run"]) == (3, 3)
    # an inf cell (below the fitted asymptote) is counted, not averaged
    mat2 = [[2.0, 8.0, math.inf]]
    fake2 = pd.DataFrame({"speedup_min": [2.0], "speedup_max": [math.inf],
                          "matrix": [mat2]}, index=["muon"])
    monkeypatch.setattr(an, "token_speedup_range", lambda study, d: fake2)
    r2 = fg.speedup_summary(df, "dense", "tokens").iloc[0]
    assert r2["mid"] == pytest.approx(4.0) and r2["n_inf"] == 1
    assert (r2["n_curve"], r2["n_run"]) == (1, 3)


def test_speedup_summary_flags_extrapolation_from_the_data(df):
    tok = fg.speedup_summary(df, "dense", "tokens").set_index("optimizer")
    assert (tok["max_budget"] == 2.0).all() and not tok["extrapolated"].any()
    d1 = _beyond_budget(df)       # Muon beats AdamW's best 2x run
    t1 = fg.speedup_summary(d1, "dense", "tokens").set_index("optimizer")
    assert (t1["max_budget"] == 2.0).all()
    assert bool(t1.loc["muon", "extrapolated"])
    assert not bool(t1.loc["manifold", "extrapolated"])
    fl = fg.speedup_summary(d1, "dense", "flops").set_index("optimizer")
    assert bool(fl.loc["muon", "extrapolated"])    # same equivalent data


def test_f6_star_is_the_accepted_value_not_the_best_cd_loss():
    rows = _two_pass_cd(inherited=False)
    # a cd cell that BEATS the accepted one (lr 0.04 at the accepted wd)
    for r in rows:
        if r["phase"] == "cd" and r["lr"] == 0.04 and r["wd"] == 0.03:
            r["final_val_loss"] = 0.55
    d = _frame(rows)
    sl = fg.cd_slices(d, "mlp256", "muon")
    fin = sl[(sl["hyper"] == "lr") & sl["final"]]
    assert fin.loc[fin["loss"].idxmin(), "value"] == 0.04       # the best cd loss
    assert list(fin[fin["accepted"]]["value"]) == [0.02]        # the star
    assert fg.accepted_config(d, "mlp256", "muon")["lr"] == 0.02
    # delta is against the optimizer's best completed cd loss (0.55), in
    # every slice — not against each slice's own minimum
    ok = sl[sl["status"] == "completed"]
    assert (ok["delta"] - (ok["loss"] - 0.55)).abs().max() < 1e-12
    assert fin[fin["accepted"]]["delta"].iloc[0] == pytest.approx(0.05)
    mo = sl[sl["hyper"] == "momentum"]
    assert mo["delta"].min() == pytest.approx(0.05)             # slice min 0.60
    wd = sl[sl["hyper"] == "wd"]
    assert sorted(wd["delta"]) == pytest.approx([0.10, 0.15, 0.17])


def test_faithful_and_halflr_rows_pair_with_the_same_seed():
    rows = _two_pass_cd(inherited=False)        # fresh: seed 1 .61, 2 .60, 3 .62
    rows.append(_row("mlp256", "muon", "faithful", 3, 0.02, 0.70, wd=0.03))
    rows.append(_row("mlp256", "muon", "faithful", 7, 0.02, 0.71, wd=0.03))  # no pair
    for seed, loss in ((1, 0.50), (2, 0.51), (3, 0.52)):
        rows.append(_row("mlp256", "adamw", "budget_constant", seed, 0.001, loss,
                         mult=2.0))
    rows.append(_row("mlp256", "adamw", "budget_halflr", 2, 0.0005, 0.53, mult=2.0))
    rows.append(_row("mlp256", "adamw", "budget_halflr", 9, 0.0005, 0.54, mult=2.0))
    d = _frame(rows)
    fa = fg.faithful_pairs(d)
    assert list(fa["seed"]) == [3]                              # seed 7: no partner
    assert fa.iloc[0]["standard"] == pytest.approx(0.62)        # seed 3, not seed 1
    assert fa.iloc[0]["faithful"] == pytest.approx(0.70)
    hl = fg.halflr_pairs(d)
    assert list(hl["seed"]) == [2]
    assert (hl.iloc[0]["full_lr"], hl.iloc[0]["half_lr"]) == (0.51, 0.53)


def test_attn_logit_bands_exclude_runs_without_a_completed_result(df, logs):
    full = fg.attn_logit_bands(logs, df, "dense")
    assert set(full["n_seeds"]) == {3}
    fresh = an._fresh(df, "dense", "muon").sort_values("seed")
    victim = fresh.iloc[0]
    for status in ("stopped", "diverged"):
        d2 = df.copy()
        d2.loc[d2["hash"] == victim["hash"], "status"] = status
        at = fg.attn_logit_bands(logs, d2, "dense")
        assert set(at[at["optimizer"] == "muon"]["n_seeds"]) == {2}
        assert set(at[at["optimizer"] == "adamw"]["n_seeds"]) == {3}
    # the result JSON missing altogether: its log is still on disk
    d3 = df[df["hash"] != victim["hash"]]
    at3 = fg.attn_logit_bands(logs, d3, "dense")
    assert set(at3[at3["optimizer"] == "muon"]["n_seeds"]) == {2}
    ev = logs[(logs["t"] == "eval") & (logs["hash"].isin(set(fresh["hash"]) -
                                                         {victim["hash"]}))]
    last = at3[at3["optimizer"] == "muon"].sort_values("x").iloc[-1]
    kept = ev[ev["images_or_tokens"] == last["x"]]["max_attn_logit"]
    assert last["mean"] == pytest.approx(kept.mean())


def test_seed_counts_come_from_the_data(df):
    assert fg.seed_counts(df, "dense") == {o: 3 for o in BASE_ORDER}
    assert list(fg.seed_counts(df, "dense")) == BASE_ORDER
    victim = an._fresh(df, "dense", "muon").iloc[0]["hash"]
    d2 = df[df["hash"] != victim]
    assert fg.seed_counts(d2, "dense")["muon"] == 2
    fg._NOTES.clear()
    assert fg.seeds_text("fx", {"a": 3, "b": 3}) == "3"
    assert fg._NOTES == []
    assert fg.seeds_text("fx", {"dense/adamw": 3, "dense/muon": 2}) == "2–3"
    assert len(fg._NOTES) == 1 and "dense/muon (2)" in fg._NOTES[0]
    assert "dense/adamw" not in fg._NOTES[0]
    fg._NOTES.clear()


def test_uneven_seed_counts_reach_the_legend_and_stdout(df, logs, tmp_path):
    victim = an._fresh(df, "dense", "muon").iloc[0]["hash"]
    d2 = df[df["hash"] != victim]
    out = io.StringIO()
    fg.build_all(d2, logs, tmp_path, out=out)
    lines = {l.split(":")[0]: l for l in out.getvalue().splitlines()}
    assert "seed counts differ" in lines["f1_final_loss"]
    assert "dense/muon (2)" in lines["f1_final_loss"]
    out3 = io.StringIO()
    fg.build_all(df, logs, tmp_path, out=out3)
    assert "seed counts differ" not in out3.getvalue()
    # the legend text itself is computed, not hard-coded
    with fg.style():
        pass
    fig_texts = []
    orig = fg._save
    try:
        def spy(fig, outdir, stem):
            fig_texts.extend(t.get_text() for lg in fig.legends for t in lg.get_texts())
            return orig(fig, outdir, stem)
        fg._save = spy
        fg.fig_f1(d2, logs, tmp_path)
    finally:
        fg._save = orig
    assert any("mean of 2–3 seeds" in t for t in fig_texts)


# ------------------------------------- Lion-Muon, the sixth optimizer (#15)

LION_BEST = curve("mlp128", LION_PLANTED["mlp128"] * D1["mlp128"])
LION_LRS = [BEST_LR["lionmuon"] * 2 ** (i - 2) for i in range(6)]
SIX_LABELS = [fg.OPT_LABEL[o] for o in fg.OPT_ORDER]
FIVE_LABELS = [fg.OPT_LABEL[o] for o in BASE_ORDER]


@pytest.fixture
def captured(monkeypatch):
    """Figures handed to ``fg._save`` (kept open for inspection, not written)."""
    figs = {}
    monkeypatch.setattr(fg, "_save", lambda fig, outdir, stem: figs.setdefault(stem, fig)
                        and [])
    yield figs
    plt.close("all")


def _legend_labels(fig) -> list[str]:
    out = [t.get_text() for lg in fig.legends for t in lg.get_texts()]
    for ax in fig.axes:
        if ax.get_legend() is not None:
            out += [t.get_text() for t in ax.get_legend().get_texts()]
    return out


def _lines_with_x(ax, xs):
    """Lines of ``ax`` whose x data are exactly ``xs``."""
    return [l for l in ax.get_lines()
            if len(l.get_xdata()) == len(xs)
            and np.allclose(np.asarray(l.get_xdata(), dtype=float), xs)]


def test_hypers_read_the_lionmuon_columns(lion_df):
    fresh = an._fresh(lion_df, "mlp128", "lionmuon").iloc[0].to_dict()
    assert fg._hypers(fresh, "lionmuon") == {
        "lr": BEST_LR["lionmuon"], "wd": 0.03, "alpha": LION_ACCEPTED["alpha"],
        "momentum": None, "beta2": LION_ACCEPTED["beta2"], "clip": math.inf}
    assert fg.accepted_config(lion_df, "mlp128", "lionmuon") == fg._hypers(
        fresh, "lionmuon")
    # the same row read as another optimizer has neither alpha nor beta2,
    # and a non-lionmuon row never picks the columns up
    assert fg._hypers(fresh, "muon")["alpha"] is None
    assert fg._hypers(fresh, "muon")["beta2"] is None
    muon = an._fresh(lion_df, "mlp128", "muon").iloc[0].to_dict()
    h = fg._hypers(muon, "muon")
    assert h["alpha"] is None and h["beta2"] is None
    assert list(h) == list(fg.CD_HYPERS)


def test_cd_slices_lionmuon_alpha_and_beta2(lion_df):
    sl = fg.cd_slices(lion_df, "mlp128", "lionmuon")
    centre = BEST_LR["lionmuon"]
    # alpha: one sweep — the four alpha-labelled cells plus alpha 30, which
    # is the wd-labelled winner it was centred on.  It ran at beta2 0.99,
    # before beta2 0.95 was accepted, so it is an earlier-pass slice.
    al = sl[sl["hyper"] == "alpha"].sort_values("value")
    assert list(al["value"]) == [3.0, 10.0, 30.0, 100.0, 300.0]
    offs = dict(LION_ALPHA_OFF)
    offs[30.0] = LION_WD_OFF[0.03]
    assert list(al["loss"]) == pytest.approx([LION_BEST + offs[v] for v in al["value"]])
    assert list(al["delta"]) == pytest.approx([offs[v] for v in al["value"]])
    assert al["slice"].nunique() == 1 and not al["final"].any()
    assert not al["accepted"].any()
    # beta2: Lion-Muon's EMA decay from config["lionmuon"]["beta2"]; 0.99 is
    # the alpha-labelled winner; the slice sits at the accepted lr/wd/alpha
    b2 = sl[sl["hyper"] == "beta2"].sort_values("value")
    assert list(b2["value"]) == [0.9, 0.95, 0.99]
    assert list(b2["delta"]) == pytest.approx(
        [LION_BETA2_OFF[0.9], LION_BETA2_OFF[0.95], LION_ALPHA_OFF[100.0]])
    assert b2["final"].all() and list(b2[b2["accepted"]]["value"]) == [0.95]
    # no Nesterov momentum: no momentum slice at all
    assert sl[sl["hyper"] == "momentum"].empty
    # lr: the accepted-context sweep (final, centre labelled "beta2") and
    # the pass-1 sweep at the defaults (earlier pass)
    lr = sl[sl["hyper"] == "lr"]
    fin, early = lr[lr["final"]].sort_values("value"), lr[~lr["final"]]
    assert list(fin["value"]) == pytest.approx(LION_LRS)
    assert list(fin["delta"]) == pytest.approx(STAGE_A_OFF["mlp128"])
    assert list(fin[fin["accepted"]]["value"]) == pytest.approx([centre])
    assert sorted(early["value"]) == pytest.approx(LION_LRS)
    assert early["delta"].min() == pytest.approx(0.05)          # the pass-1 handicap
    # wd: swept at the default alpha / beta2 -> an earlier pass
    wd = sl[sl["hyper"] == "wd"]
    assert sorted(wd["value"]) == [0.0, 0.003, 0.01, 0.03] and not wd["final"].any()
    assert set(sl["hyper"]) == {"lr", "wd", "alpha", "beta2"}


def test_cd_slices_of_the_other_optimizers_unchanged_by_lionmuon(df, lion_df):
    for opt in WORLD_OPTS:
        a, b = fg.cd_slices(df, "mlp128", opt), fg.cd_slices(lion_df, "mlp128", opt)
        assert not a.empty
        pd.testing.assert_frame_equal(a, b)
        assert "alpha" not in set(a["hyper"])
        pd.testing.assert_frame_equal(fg.lr_sweep(df, "mlp128", opt),
                                      fg.lr_sweep(lion_df, "mlp128", opt))


def test_lr_sweep_and_stability_for_lionmuon(lion_df):
    sw = fg.lr_sweep(lion_df, "mlp128", "lionmuon")
    assert list(sw["lr"]) == pytest.approx(LION_LRS)
    assert list(sw["loss"]) == pytest.approx(
        [LION_BEST + off for off in STAGE_A_OFF["mlp128"]])    # not the pass-1 losses
    assert (sw["status"] == "completed").all()
    assert fg.earlier_pass_divergences(lion_df, "mlp128", "lionmuon") == []
    st = fg.lr_stability(lion_df, "mlp128", "lionmuon")
    assert list(st["lr"]) == pytest.approx(LION_LRS) and (st["state"] == "ok").all()
    assert list(st[st["best"]]["lr"]) == pytest.approx([BEST_LR["lionmuon"]])
    # dense is Stage-A tuned in this world: the stage_a + stage_b lrs
    assert len(fg.lr_sweep(lion_df, "dense", "lionmuon")) == 6


def test_headline_helpers_include_lionmuon(df, lion_df, lion_logs):
    runs = fg.headline_runs(lion_df, "dense")
    assert len(runs) == 18 and list(runs["optimizer"].unique()) == list(fg.OPT_ORDER)
    assert fg.present_optimizers(lion_df, "dense") == list(fg.OPT_ORDER)
    assert fg.present_optimizers(df, "dense") == BASE_ORDER
    assert fg.present_optimizers(lion_df, "moe") == BASE_ORDER    # no MoE Lion-Muon
    assert fg.seed_counts(lion_df, "mlp128")["lionmuon"] == 3
    # speedup: the planted value, in the house order
    for study, s in LION_PLANTED.items():
        tok = fg.speedup_summary(lion_df, study, "tokens").set_index("optimizer")
        assert list(tok.index) == list(fg.OPT_ORDER)
        assert tok.loc["lionmuon", "mid"] == pytest.approx(s, rel=0.02)
        assert not bool(tok.loc["lionmuon", "extrapolated"])
        fl = fg.speedup_summary(lion_df, study, "flops").set_index("optimizer")
        assert fl.loc["lionmuon", "mid"] == pytest.approx(
            s * (1 + OPT_RATE["adamw"]) / (1 + LION_OPT_RATE), rel=0.02)
    ext = fg.speedup_summary(_beyond_budget(lion_df, "dense", "lionmuon"), "dense")
    assert bool(ext.set_index("optimizer").loc["lionmuon", "extrapolated"])
    # F7: its optimizer FLOPs are the measured flops_optimizer, like the others
    c = fg.compute_cost(lion_df, "dense").set_index("optimizer")
    assert list(c.index) == list(fg.OPT_ORDER)
    assert c.loc["lionmuon", "optimizer_cost"] == pytest.approx(LION_OPT_RATE)
    assert c.loc["lionmuon", "total"] == pytest.approx(1 + LION_OPT_RATE)
    fr = an._fresh(lion_df, "dense", "lionmuon")
    assert c.loc["lionmuon", "optimizer_cost"] == pytest.approx(
        float((fr["flops_optimizer"] / fr["flops_model"]).mean()))
    # curves, attention logits, spectral norms
    cur = fg.loss_curves(lion_df, lion_logs, "dense", "tokens")
    g = cur[cur["optimizer"] == "lionmuon"].sort_values("k")
    assert (g["n_seeds"] == 3).all() and g["x"].iloc[-1] == pytest.approx(D1["dense"])
    ev = lion_logs[(lion_logs["study"] == "dense") & (lion_logs["t"] == "eval")
                   & (lion_logs["optimizer"] == "lionmuon")]
    last = ev[ev["step"] == ev["step"].max()]["val_loss"]
    assert len(last) == 3 and g["mean"].iloc[-1] == pytest.approx(last.mean())
    fl = fg.loss_curves(lion_df, lion_logs, "dense", "flops")
    gf = fl[fl["optimizer"] == "lionmuon"].sort_values("k")
    assert gf["x"].iloc[-1] == pytest.approx(float(fr["flops_total"].mean()))
    assert "lionmuon" in set(fg.attn_logit_bands(lion_logs, lion_df, "dense")["optimizer"])
    assert fg.spectral_seed_counts(lion_logs, lion_df, "dense")["lionmuon"] == 3
    assert "lionmuon" in set(fg.spectral_by_layer(lion_logs, lion_df, "dense")["optimizer"])
    # no inherited track, no faithful rerun, no init ablation for it
    assert "lionmuon" not in set(fg.track_comparison(lion_df)["optimizer"])
    assert "lionmuon" not in set(fg.faithful_pairs(lion_df)["optimizer"])


def test_rank_table_ranks_six(df, lion_df):
    tab = fg.rank_table(lion_df)
    for study in ("mlp128", "dense"):
        d = tab[tab["study"] == study].set_index("optimizer")
        assert sorted(d["rank"]) == [1, 2, 3, 4, 5, 6]
        # planted best in both studies; its mean is the fresh-seed mean
        assert d.loc["lionmuon", "rank"] == 1
        assert d.loc["lionmuon", "mean_loss"] == pytest.approx(
            an._fresh(lion_df, study, "lionmuon")["final_val_loss"].mean())
        # every other optimizer moves down exactly one rank, same order
        old = fg.rank_table(df)
        old = old[old["study"] == study].set_index("optimizer")
        for opt in WORLD_OPTS:
            assert d.loc[opt, "rank"] == old.loc[opt, "rank"] + 1
        # the old winner is now compared with Lion-Muon by the seed rule:
        # a clear planted gap with a 2e-5 seed spread -> separated
        first_old = old[old["rank"] == 1].index[0]
        assert bool(d.loc[first_old, "separated"]) is True
        pair = an._paired_vs(lion_df, study, first_old, "lionmuon")
        assert an.seed_rule(*pair)
    # ... and the rule can fail against it: seeds that straddle the old winner
    d2 = lion_df.copy()
    m = ((d2["study"] == "dense") & (d2["optimizer"] == "lionmuon")
         & (d2["phase"] == "fresh_seed"))
    muon = an._fresh(lion_df, "dense", "muon").sort_values("seed")["final_val_loss"]
    d2.loc[m, "final_val_loss"] = (muon.to_numpy() + np.array([0.02, -0.03, -0.02]))
    t2 = fg.rank_table(d2)
    t2 = t2[t2["study"] == "dense"].set_index("optimizer")
    assert t2.loc["lionmuon", "rank"] == 1 and bool(t2.loc["muon", "separated"]) is False


def test_f6_columns():
    five = ["lr", "wd", "momentum", "beta2", "clip"]
    assert fg.f6_columns(BASE_ORDER) == five                 # the grid before Lion-Muon
    assert fg.f6_columns(["adamw"]) == five
    assert fg.f6_columns(list(fg.OPT_ORDER)) == ["lr", "wd", "alpha", "momentum",
                                                 "beta2", "clip"]
    assert fg.f6_columns(["lionmuon"])[2] == "alpha"


def test_f6_grid_without_and_with_lionmuon(df, lion_df, captured):
    fg.fig_f6(df, None, None, "mlp128")
    base = captured.pop("f6_hparam_sensitivity_mlp128")
    assert len(base.axes) == 5 * 5
    assert [ax.get_title() for ax in base.axes[:5]] == [
        fg.CD_HYPER_LABEL[h] for h in ("lr", "wd", "momentum", "beta2", "clip")]
    assert [base.axes[5 * i].get_ylabel() for i in range(5)] == FIVE_LABELS
    assert not any("α" in ax.get_title() for ax in base.axes)

    fg.fig_f6(lion_df, None, None, "mlp128")
    fig = captured.pop("f6_hparam_sensitivity_mlp128")
    assert len(fig.axes) == 6 * 6
    assert [ax.get_title() for ax in fig.axes[:6]] == [
        fg.CD_HYPER_LABEL_NARROW[h] for h in fg.CD_HYPERS]
    assert [fig.axes[6 * i].get_ylabel() for i in range(6)] == SIX_LABELS
    cell = lambda opt, hyper: fig.axes[6 * fg.OPT_ORDER.index(opt)       # noqa: E731
                                       + fg.CD_HYPERS.index(hyper)]
    na = lambda ax: [t.get_text() for t in ax.texts] == ["n/a"]          # noqa: E731
    # alpha: n/a for everyone but Lion-Muon; Lion-Muon has no momentum cell
    for opt in WORLD_OPTS:
        assert na(cell(opt, "alpha")), opt
    assert na(cell("lionmuon", "momentum"))
    assert not na(cell("muon", "wd")) and not na(cell("lionmuon", "wd"))
    assert na(cell("muon", "beta2"))
    # the alpha sweep: the tested values on a log axis, losses minus the best
    ax = cell("lionmuon", "alpha")
    assert not na(ax) and ax.get_xscale() == "log"
    offs = dict(LION_ALPHA_OFF)
    offs[30.0] = LION_WD_OFF[0.03]
    alphas = sorted(offs)
    line = _lines_with_x(ax, alphas)[0]
    assert list(line.get_ydata()) == pytest.approx([offs[a] for a in alphas])
    assert line.get_color() == fg.OPT_COLOR["lionmuon"]
    ticks = sorted(list(ax.xaxis.get_majorticklocs()) + list(ax.xaxis.get_minorticklocs()))
    assert ticks == pytest.approx(alphas)                    # every value ticked
    fig.canvas.draw()
    assert [t.get_text() for t in ax.get_xticklabels()] == ["3", "30", "300"]
    # the beta2 sweep sits in the beta2 column, star on the accepted 0.95
    ax = cell("lionmuon", "beta2")
    assert [t.get_text() for t in ax.get_xticklabels()] == [".9", ".95", ".99"]
    line = _lines_with_x(ax, [0, 1, 2])[0]
    assert list(line.get_ydata()) == pytest.approx(
        [LION_BETA2_OFF[0.9], LION_BETA2_OFF[0.95], LION_ALPHA_OFF[100.0]])
    stars = [l for l in ax.get_lines() if l.get_marker() == "*"]
    assert len(stars) == 1
    assert list(stars[0].get_xdata()) == [1] and list(stars[0].get_ydata()) == [0.0]


def _six_width_frame():
    rows = _width_frame().to_dict("records")
    for study in ("mlp128", "mlp256", "mlp512"):
        for opt in ("moonlight", "muonall", "lionmuon", "manifold"):
            olr = 0.002 if opt == "lionmuon" and study == "mlp512" else 0.004
            for seed in (1, 2, 3):
                rows.append(_row(study, opt, "fresh_seed", seed, olr, 0.7 + 0.01 * seed,
                                 alpha=100.0, beta2=0.95))
            for k, off in ((0.5, 0.03), (1, 0.0), (2, 0.02)):
                rows.append(_row(study, opt, "cd", 0, olr * k, 0.7 + off, cd_sweep="lr",
                                 alpha=100.0, beta2=0.95))
    return _frame(rows)


def test_f5_one_block_for_five_two_rows_of_three_for_six(captured):
    six = _six_width_frame()
    five = six[six["optimizer"] != "lionmuon"]
    fg.fig_f5(five, None, None)
    fig = captured.pop("f5_width_transfer")
    assert len(fig.axes) == 10
    assert fig.axes[0].get_subplotspec().get_gridspec().get_geometry() == (2, 5)
    assert [ax.get_title() for ax in fig.axes[:5]] == FIVE_LABELS
    assert fig.get_size_inches()[0] == pytest.approx(fg.FULL_W)

    fg.fig_f5(six, None, None)
    fig = captured.pop("f5_width_transfer")
    assert fig.axes[0].get_subplotspec().get_gridspec().get_geometry() == (4, 3)
    assert len(fig.axes) == 12 and all(ax.get_visible() for ax in fig.axes)
    assert fig.get_size_inches()[0] == pytest.approx(fg.FULL_W)
    assert [ax.get_title() for ax in fig.axes[0:3]] + [
        ax.get_title() for ax in fig.axes[6:9]] == SIX_LABELS
    # the Lion-Muon panel: one curve per width, at its own learning rates,
    # and the shift of its selected LR from 128 to 512 (0.004 -> 0.002)
    top, bottom = fig.axes[7], fig.axes[10]
    assert top.get_title() == "Lion-Muon" and bottom.get_title() == "128→512: ÷2"
    for w, olr in ((128, 0.004), (256, 0.004), (512, 0.002)):
        xs = [olr * 0.5, olr, olr * 2]
        ln = [l for l in _lines_with_x(top, xs) if l.get_color() == fg.WIDTH_COLOR[w]]
        assert len(ln) == 1
        assert list(ln[0].get_ydata()) == pytest.approx([0.73, 0.70, 0.72])
        lb = [l for l in _lines_with_x(bottom, xs) if l.get_color() == fg.WIDTH_COLOR[w]]
        assert list(lb[0].get_ydata()) == pytest.approx([0.03, 0.0, 0.02])
    # x is shared within a panel pair, the basin y axis across all panels
    assert top.get_xlim() == bottom.get_xlim()
    assert len({fig.axes[i].get_ylim() for i in (3, 4, 5, 9, 10, 11)}) == 1
    assert fig.axes[0].get_ylabel() and fig.axes[6].get_ylabel()
    assert not fig.axes[7].get_ylabel()


def test_f5b_six_panels_in_one_row():
    d = _six_width_frame()
    fig = fg.build_f5b(d)
    try:
        assert [ax.get_title() for ax in fig.axes] == SIX_LABELS
        assert fig.axes[0].get_subplotspec().get_gridspec().get_geometry() == (1, 6)
        ax = fig.axes[fg.OPT_ORDER.index("lionmuon")]
        lines = [l for l in ax.get_lines() if l.get_gid() in ("retuned", "inherited")]
        assert [l.get_gid() for l in lines] == ["retuned"]      # no inherited track
        assert [float(v) for v in lines[0].get_xdata()] == [128.0, 256.0, 512.0]
        assert list(lines[0].get_ydata()) == pytest.approx([0.004, 0.004, 0.002])
        assert lines[0].get_color() == "#6b3fa0" and lines[0].get_marker() == "P"
    finally:
        plt.close(fig)


def test_legends_and_rows_hold_lionmuon_only_when_it_has_data(
        df, logs, lion_df, lion_logs, captured):
    legend_figs = {"f3_budget_curve": fg.fig_f3, "f4_loss_curves_tokens": fg.fig_f4,
                   "f4b_loss_curves_flops": fg.fig_f4b, "a1_lr_sensitivity": fg.fig_a1,
                   "a4_loss_vs_accuracy": fg.fig_a4, "a6_attention_logits": fg.fig_a6}
    row_figs = {"f1_final_loss": fg.fig_f1, "f2_speedup": fg.fig_f2,
                "f7_compute_cost": fg.fig_f7, "a8_divergence": fg.fig_a8}
    for stem, fn in {**legend_figs, **row_figs}.items():
        fn(df, logs, None)
        base = captured.pop(stem)
        fn(lion_df, lion_logs, None)
        full = captured.pop(stem)
        if stem in legend_figs:
            assert not any("Lion-Muon" in t for t in _legend_labels(base)), stem
            got = [t for t in _legend_labels(full) if "Lion-Muon" in t]
            assert len(got) == 1, (stem, _legend_labels(full))
            # the legend lists the optimizers in the house order
            names = [t.replace(" at 1×", "") for t in _legend_labels(full)]
            names = [t for t in names if t in SIX_LABELS]
            assert names == [n for n in SIX_LABELS if n in names], stem
        else:
            rows = lambda fig: [t.get_text() for t in fig.axes[0].get_yticklabels()]  # noqa: E731
            assert len(base.axes[0].get_yticks()) == 5, stem
            assert len(full.axes[0].get_yticks()) == 6, stem
            if stem != "f2_speedup":    # F2's shared y axis prints no row names
                assert rows(base) == FIVE_LABELS, stem
                assert rows(full) == SIX_LABELS, stem
        plt.close(base)
        plt.close(full)
    # F8: direct labels instead of a legend
    fg.fig_f8(df, logs, None)
    texts = [t.get_text() for t in captured.pop("f8_rank_summary").axes[0].texts]
    assert sorted(texts) == sorted(FIVE_LABELS)
    fg.fig_f8(lion_df, lion_logs, None)
    ax = captured.pop("f8_rank_summary").axes[0]
    assert sorted(t.get_text() for t in ax.texts) == sorted(SIX_LABELS)
    assert [t.get_text() for t in ax.get_yticklabels()] == [
        "1 (best)", "2", "3", "4", "5", "6"]


def test_a_study_without_lionmuon_runs_omits_it(lion_df, lion_logs, captured, tmp_path):
    # Lion-Muon results exist for mlp128 only (the other studies still running)
    part = lion_df[~((lion_df["optimizer"] == "lionmuon") & (lion_df["study"] == "dense"))]
    assert fg.present_optimizers(part, "dense") == BASE_ORDER
    assert "lionmuon" not in set(fg.rank_table(part).query("study == 'dense'")["optimizer"])
    assert sorted(fg.rank_table(part).query("study == 'dense'")["rank"]) == [1, 2, 3, 4, 5]
    assert "lionmuon" not in set(fg.speedup_summary(part, "dense")["optimizer"])
    assert "lionmuon" not in set(fg.compute_cost(part, "dense")["optimizer"])
    assert fg.lr_stability(part, "dense", "lionmuon").empty
    # A8 labels every panel: the dense panel has no Lion-Muon row at all
    fg.fig_a8(part, lion_logs, None)
    fig = captured.pop("a8_divergence")
    rows = {ax.get_title(): [t.get_text() for t in ax.get_yticklabels()]
            for ax in fig.axes if ax.get_visible()}
    assert rows == {"MLP-128": SIX_LABELS, "Dense transformer": FIVE_LABELS}
    # curves of the dense panel: five optimizer colours, no Lion-Muon line
    fg.fig_f4(part, lion_logs, None)
    fig = captured.pop("f4_loss_curves_tokens")
    by_title = {ax.get_title(): ax for ax in fig.axes if ax.get_title()}
    colours = lambda ax: {l.get_color() for l in ax.get_lines()}       # noqa: E731
    assert fg.OPT_COLOR["lionmuon"] in colours(by_title["MLP-128"])
    assert fg.OPT_COLOR["lionmuon"] not in colours(by_title["Dense transformer"])
    assert colours(by_title["Dense transformer"]) == {fg.OPT_COLOR[o] for o in BASE_ORDER}


def test_every_figure_renders_with_lionmuon(df, logs, lion_df, lion_logs, tmp_path):
    res = {}
    for name, (d, l) in {"base": (df, logs), "lion": (lion_df, lion_logs)}.items():
        buf = io.StringIO()
        res[name] = fg.build_all(d, l, tmp_path / name, out=buf)
        assert "seed counts differ" not in buf.getvalue()
        for fid, paths in res[name].items():
            for p in paths:
                assert p.exists() and p.stat().st_size > 1000, (name, fid)
    # the same figures are written (and skipped) with and without Lion-Muon
    assert {k for k, v in res["base"].items() if v} == {
        k for k, v in res["lion"].items() if v}
    assert "f6_hparam_sensitivity_mlp128" in res["lion"]
    # partial data (Lion-Muon in one study only) renders too
    part = lion_df[~((lion_df["optimizer"] == "lionmuon") & (lion_df["study"] == "dense"))]
    out = fg.build_all(part, lion_logs, tmp_path / "part", out=io.StringIO())
    assert {k for k, v in out.items() if v} == {k for k, v in res["lion"].items() if v}
    assert plt.get_fignums() == []


def test_hand_built_frame_with_lionmuon_renders(tmp_path):
    # the two-pass CD frame for muon, plus a Lion-Muon CD with alpha and
    # beta2 sweeps, pilots — and no logs at all
    rows = _two_pass_cd(inherited=True)
    kw = dict(wd=0.03, alpha=100.0, beta2=0.95)
    for lr, loss in ((0.0005, 0.70), (0.001, 0.62), (0.002, 0.66)):
        rows.append(_row("mlp256", "lionmuon", "cd", 0, lr, loss, cd_sweep="lr", **kw))
    for alpha, loss in ((3.0, 0.69), (10.0, 0.65), (30.0, 0.63), (300.0, 0.64)):
        rows.append(_row("mlp256", "lionmuon", "cd", 0, 0.001, loss, cd_sweep="alpha",
                         **{**kw, "alpha": alpha}))
    for beta2, loss in ((0.9, 0.64), (0.99, 0.63)):
        rows.append(_row("mlp256", "lionmuon", "cd", 0, 0.001, loss, cd_sweep="beta2",
                         **{**kw, "beta2": beta2}))
    rows.append(_row("mlp256", "lionmuon", "cd", 0, 0.004, None, cd_sweep="lr",
                     status="diverged", **kw))
    for seed, loss in ((1, 0.62), (2, 0.63), (3, 0.61)):
        rows.append(_row("mlp256", "lionmuon", "fresh_seed", seed, 0.001, loss, **kw))
    rows += [_row("mlp256", "adamw", "pilot", s, 0.001, l)
             for s, l in ((101, 0.80), (102, 0.81), (103, 0.82))]
    d = _frame(rows)
    sl = fg.cd_slices(d, "mlp256", "lionmuon")
    al = sl[sl["hyper"] == "alpha"]
    assert sorted(al["value"]) == [3.0, 10.0, 30.0, 100.0, 300.0] and al["final"].all()
    assert list(al[al["accepted"]]["value"]) == [100.0]
    b2 = sl[sl["hyper"] == "beta2"]
    assert sorted(b2["value"]) == [0.9, 0.95, 0.99]
    assert list(b2[b2["accepted"]]["value"]) == [0.95]
    lr = sl[(sl["hyper"] == "lr") & sl["final"]]
    assert list(lr[lr["status"] == "diverged"]["value"]) == [0.004]
    assert list(fg.lr_sweep(d, "mlp256", "lionmuon")["lr"]) == [0.0005, 0.001, 0.002,
                                                               0.004]
    res = fg.build_all(d, None, tmp_path, out=io.StringIO())
    for fid in ("f1_final_loss", "f5_width_transfer", "f5b_optimal_lr_vs_width",
                "f6_hparam_sensitivity_mlp256", "f7_compute_cost", "f8_rank_summary",
                "a1_lr_sensitivity", "a2_inherited_vs_retuned", "a8_divergence"):
        assert res[fid], fid
    assert plt.get_fignums() == []
