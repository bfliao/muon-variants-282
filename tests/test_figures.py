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
from tests.test_analysis import D1, OPT_RATE, PLANTED, make_world

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


_COUNTER = [0]


def _row(study, opt, phase, seed, lr, loss, *, wd=0.01, momentum=0.95,
         betas=(0.9, 0.95), clip=1.0, status="completed", mult=1.0,
         cd_sweep=None, width=None, fm=100.0, fo=10.0):
    _COUNTER[0] += 1
    if width is None and study.startswith("mlp"):
        width = int(study.replace("mlp", ""))
    return {
        "study": study, "optimizer": opt, "phase": phase, "seed": seed, "lr": lr,
        "wd": wd, "momentum": None if opt == "adamw" else momentum,
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
              "budget_multiplier"):
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
    assert fg.OPT_ORDER == ("adamw", "muon", "moonlight", "muonall", "manifold")
    assert fg.OPT_LABEL == {"adamw": "AdamW", "muon": "Muon", "moonlight": "Moonlight",
                            "muonall": "MuonAll", "manifold": "Manifold Muon"}
    assert fg.OPT_COLOR == {"adamw": "#3a3a38", "muon": "#2a78d6",
                            "moonlight": "#eda100", "manifold": "#008300",
                            "muonall": "#e87ba4"}
    assert fg.OPT_MARKER == {"adamw": "o", "muon": "s", "moonlight": "D",
                             "manifold": "^", "muonall": "v"}
    # every optimizer the analysis knows is styled, and nothing else
    for table in (fg.OPT_LABEL, fg.OPT_COLOR, fg.OPT_MARKER):
        assert set(table) == set(an.OPTIMIZERS) == set(fg.OPT_ORDER)
    assert len(set(fg.OPT_COLOR.values())) == 5
    assert len(set(fg.OPT_MARKER.values())) == 5
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
    assert list(runs["optimizer"].unique()) == list(fg.OPT_ORDER)
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
    assert acc == {"lr": 0.02, "wd": 0.03, "momentum": 0.95, "beta2": None,
                   "clip": 1.0}
    assert fg.accepted_config(d, "mlp256", "adamw") is None
    # AdamW: momentum means beta1, beta2 exists, no clipping -> +inf
    a = _frame([_row("mlp128", "adamw", "fresh_seed", 1, 0.001, 0.9,
                     betas=(0.8, 0.99), clip=None)])
    assert fg.accepted_config(a, "mlp128", "adamw") == {
        "lr": 0.001, "wd": 0.01, "momentum": 0.8, "beta2": 0.99, "clip": math.inf}


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
    assert set(tok["optimizer"]) == set(an.OPTIMIZERS)
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
