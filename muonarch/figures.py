"""Report figures (docs/figures_spec.md, v2).

Three sections:

* **style** — the house palette, markers, labels and the rcParams context
  manager every figure is drawn in;
* **data helpers** — small pure functions that turn the results / logs frames
  of ``muonarch.analysis`` into the tidy frames each figure plots (tested in
  tests/test_figures.py);
* **figures** — one ``fig_<id>(df, logs, outdir) -> list[Path]`` per figure
  and ``main()``, which builds all of them and prints one line per figure.

Nothing here trains, and nothing is written outside ``outdir``.  Every number
drawn comes from the result / log files; nothing is hard-coded.

Conventions that go beyond the spec text (all stated on the figures):

* The "accepted" hyperparameters of an optimizer are the ones its fresh-seed
  (headline) runs used.  The learning-rate sweep shown for an optimizer is the
  coordinate-descent slice at exactly those accepted values: ``lr_sweep`` is a
  thin view of ``analysis._lr_sweep_rows`` (the one implementation).
* The central speedup value is the geometric mean of the 3x3
  {AdamW curve seed} x {optimizer seed} matrix; the whisker is its min-max.
"""

from __future__ import annotations

import contextlib
import json
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.offsetbox import (AnnotationBbox, DrawingArea, HPacker, TextArea,
                                  VPacker)
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullFormatter

from muonarch import analysis as an

# ===========================================================================
# Style

#: Fixed optimizer order of every figure.
OPT_ORDER: tuple[str, ...] = ("adamw", "muon", "moonlight", "muonall", "lionmuon",
                              "manifold")

OPT_LABEL: dict[str, str] = {
    "adamw": "AdamW",
    "muon": "Muon",
    "moonlight": "Moonlight",
    "muonall": "MuonAll",
    "lionmuon": "Lion-Muon",
    "manifold": "Manifold Muon",
}
OPT_COLOR: dict[str, str] = {
    "adamw": "#3a3a38",
    "muon": "#2a78d6",
    "moonlight": "#eda100",
    "manifold": "#008300",
    "muonall": "#e87ba4",
    "lionmuon": "#6b3fa0",
}
OPT_MARKER: dict[str, str] = {
    "adamw": "o",
    "muon": "s",
    "moonlight": "D",
    "manifold": "^",
    "muonall": "v",
    "lionmuon": "P",
}
STUDY_LABEL: dict[str, str] = {
    "mlp128": "MLP-128",
    "mlp256": "MLP-256",
    "mlp512": "MLP-512",
    "dense": "Dense transformer",
    "moe": "MoE transformer",
}
#: The studies of the headline figures, in display order.  MoE joins only
#: once its fresh seeds exist (its figures are skipped until then).
HEADLINE_STUDIES: tuple[str, ...] = ("mlp128", "mlp256", "mlp512", "dense")
MLP_STUDIES: tuple[str, ...] = ("mlp128", "mlp256", "mlp512")
WIDTH_COLOR: dict[int, str] = {128: "#86b6ef", 256: "#2a78d6", 512: "#0d366b"}

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e0"
NEUTRAL = "#d9d8d2"      # neutral fill (model FLOPs, reference bands)
SHADE = "#f3f2ee"        # very light region shading
PALE = "#a3a29c"         # neutral marks (per-seed AdamW points, n/a text)

FULL_W = 7.2
COL_W = 3.5
DPI = 200

#: CD hyperparameter columns of F6, in the sweep order of studies.CD_ORDER.
#: "momentum" is beta1 for AdamW (as in studies.cd_hypers); "beta2" is AdamW's
#: second-moment decay and Lion-Muon's EMA decay; "alpha" (Muon-step lr ÷
#: sign-step lr) exists for Lion-Muon only.
CD_HYPERS: tuple[str, ...] = ("lr", "wd", "alpha", "momentum", "beta2", "clip")
CD_HYPER_LABEL: dict[str, str] = {
    "lr": "Learning rate",
    "wd": "Weight decay",
    "alpha": "α (Muon LR ÷ sign LR)",
    "momentum": "Momentum / β₁",
    "beta2": "β₂",
    "clip": "Gradient clip",
}
#: Columns F6 draws only when the optimizer that owns them is in the figure.
CD_HYPER_OWNER: dict[str, str] = {"alpha": "lionmuon"}
#: Two-line column titles for the six-column F6 grid (a column is then about
#: 1 in wide, too narrow for the one-line titles).
CD_HYPER_LABEL_NARROW: dict[str, str] = {
    "lr": "Learning\nrate",
    "wd": "Weight\ndecay",
    "alpha": "α = Muon LR\n÷ sign LR",
    "momentum": "Momentum\n/ β₁",
    "beta2": "β₂",
    "clip": "Gradient\nclip",
}
#: CIFAR test-split size (a protocol constant, muonarch/data.py CIFAR_N_TEST),
#: used only for the binomial standard error bar of A4.
N_TEST_IMAGES = 10_000

_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.edgecolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.8,
    "lines.markersize": 6.5,
    "lines.markeredgecolor": "white",
    "lines.markeredgewidth": 1.0,
    "figure.dpi": 100,
    "savefig.dpi": DPI,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.05,
    "figure.constrained_layout.w_pad": 0.05,
}


@contextlib.contextmanager
def style():
    """rcParams context of the house style (docs/figures_spec.md)."""
    with matplotlib.rc_context(_RC):
        yield


def _opt_handle(opt: str, line: bool = True) -> Line2D:
    return Line2D([], [], color=OPT_COLOR[opt], marker=OPT_MARKER[opt],
                  linestyle="-" if line else "none", lw=1.8, ms=6.5,
                  mec="white", mew=1.0, label=OPT_LABEL[opt])


def _legend(fig, handles, ncol: int, loc: str = "outside upper center", **kw):
    """The figure's single legend, outside the plot area."""
    kw.setdefault("handlelength", 2.0)
    kw.setdefault("columnspacing", 1.4)
    kw.setdefault("handletextpad", 0.5)
    kw.setdefault("borderaxespad", 0.2)
    leg = fig.legend(handles=handles, loc=loc, ncol=ncol, **kw)
    for t in leg.get_texts():
        t.set_color(INK)
    if leg.get_title() is not None:
        leg.get_title().set_color(INK)
    return leg


def _save(fig, outdir: Path | str, stem: str) -> list[Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = [outdir / f"{stem}.png", outdir / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=DPI)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def _subplots(nrows, ncols, figsize, **kw):
    kw.setdefault("squeeze", False)
    return plt.subplots(nrows, ncols, figsize=figsize, **kw)


def fmt_mult(v: float) -> str:
    """'2.3×' / '0.64×' / '0.036×' / '∞'."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if math.isinf(v):
        return "∞"
    if v >= 9.95:
        return f"{v:.0f}×"
    if v >= 0.995:
        return f"{v:.1f}×"
    if v >= 0.0995:
        return f"{v:.2f}×"
    return f"{v:.3f}×"


def fmt_num(v: float) -> str:
    """Compact exact number for a tick label: 0.003, 0.0005, 3.125e-3 -> 0.003125."""
    if v is None:
        return "n/a"
    if math.isinf(v):
        return "none"
    return f"{v:.6g}"


def fmt_cat(v: float) -> str:
    """Compact categorical tick label: '.003', '.95', '1', '0', 'none'."""
    t = fmt_num(v)
    return t[1:] if t.startswith("0.") else t


def fmt_shift(ratio: float) -> str:
    """Optimum shift label from a learning-rate ratio: '÷2', '×4', 'no shift'."""
    if ratio is None or not np.isfinite(ratio) or ratio <= 0:
        return "n/a"
    if abs(math.log2(ratio)) < 1e-9:
        return "no shift"
    f = ratio if ratio > 1 else 1.0 / ratio
    return f"{'×' if ratio > 1 else '÷'}{f:.3g}"


_NOTES: list[str] = []
SKIP_REASONS: dict[str, str] = {}


def _note(msg: str) -> None:
    """A stdout note about a skipped panel / data caveat (printed by main)."""
    if msg not in _NOTES:
        _NOTES.append(msg)


def _skip(fig_id: str, reason: str) -> list[Path]:
    SKIP_REASONS[fig_id] = reason
    return []


# ===========================================================================
# Data helpers (pure; tested)

def headline_studies(df: pd.DataFrame) -> list[str]:
    """Headline studies (fixed order) that have completed fresh-seed runs."""
    if df is None or df.empty:
        return []
    return [s for s in HEADLINE_STUDIES if not an._fresh(df, s).empty]


def present_optimizers(df: pd.DataFrame, study: str) -> list[str]:
    """Optimizers with completed fresh-seed runs in the study, house order."""
    return [o for o in OPT_ORDER if not an._fresh(df, study, o).empty]


def headline_runs(df: pd.DataFrame, study: str) -> pd.DataFrame:
    """The fresh-seed 1x runs of a study: optimizer, seed, loss, status.

    Completed AND diverged runs are returned (diverged with NaN loss) so a
    figure can draw the divergence instead of dropping it; any other status
    (a stopped / partial run) is never counted.  Tuning phases — including the
    inherited stage_a / stage_b / edge track — never enter."""
    cols = ["optimizer", "seed", "loss", "status"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    m = ((df["study"] == study) & (df["phase"] == "fresh_seed")
         & df["status"].isin(["completed", "diverged"]))
    if "budget_multiplier" in df.columns:
        m &= df["budget_multiplier"] == 1.0
    r = df[m]
    out = pd.DataFrame({
        "optimizer": r["optimizer"].to_numpy(),
        "seed": r["seed"].to_numpy(),
        "loss": np.where(r["status"] == "completed",
                         r["final_val_loss"].astype(float), np.nan),
        "status": r["status"].to_numpy(),
    }, columns=cols)
    out = out[out["optimizer"].isin(OPT_ORDER)]
    out["_o"] = out["optimizer"].map(OPT_ORDER.index)
    return out.sort_values(["_o", "seed"]).drop(columns="_o").reset_index(drop=True)


def _num(v) -> float | None:
    try:
        if v is None or pd.isna(v):
            return None
    except (TypeError, ValueError):
        return None
    try:
        return float(f"{float(v):.10g}")
    except (TypeError, ValueError):
        return None


def _hypers(row: dict, opt: str) -> dict:
    """The CD hyperparameters of a run, normalised and hashable.  ``momentum``
    is beta1 for AdamW and does not exist for Lion-Muon; ``beta2`` exists for
    AdamW (``adamw_betas[1]``) and for Lion-Muon (its EMA decay, the
    ``lionmuon_beta2`` column of ``analysis.load_results``); ``alpha`` exists
    for Lion-Muon only (``lionmuon_alpha``); ``clip`` is +inf for 'no
    clipping'."""
    betas = row.get("adamw_betas")
    b1 = b2 = None
    if isinstance(betas, (list, tuple)) and len(betas) == 2:
        b1, b2 = _num(betas[0]), _num(betas[1])
    clip = _num(row.get("grad_clip"))
    if opt == "lionmuon":
        momentum, alpha = None, _num(row.get("lionmuon_alpha"))
        beta2 = _num(row.get("lionmuon_beta2"))
    else:
        momentum = b1 if opt == "adamw" else _num(row.get("momentum"))
        alpha, beta2 = None, (b2 if opt == "adamw" else None)
    return {
        "lr": _num(row.get("lr")),
        "wd": _num(row.get("wd")),
        "alpha": alpha,
        "momentum": momentum,
        "beta2": beta2,
        "clip": math.inf if clip is None else clip,
    }


def f6_columns(opts) -> list[str]:
    """The hyperparameter columns of an F6 grid over ``opts``, in CD_HYPERS
    order: a column owned by one optimizer (alpha: Lion-Muon) is drawn only
    when that optimizer is in the grid."""
    return [h for h in CD_HYPERS
            if CD_HYPER_OWNER.get(h) is None or CD_HYPER_OWNER[h] in opts]


def accepted_config(df: pd.DataFrame, study: str, opt: str) -> dict | None:
    """The accepted (selected) hyperparameters: those of the optimizer's
    completed fresh-seed runs.  None without fresh runs."""
    fr = an._fresh(df, study, opt)
    if fr.empty:
        return None
    return _hypers(fr.sort_values("seed").iloc[0].to_dict(), opt)


_SLICE_COLS = ["hyper", "value", "loss", "status", "delta", "final",
               "accepted", "slice"]


def cd_slices(df: pd.DataFrame, study: str, opt: str) -> pd.DataFrame:
    """Every one-dimensional coordinate-descent slice of (study, optimizer).

    A slice is a set of >= 2 seed-0 ``cd`` runs that differ in exactly one
    hyperparameter (``hyper``) and agree on all others, at least one of which
    was run AS a sweep of that hyperparameter (its ``cd_sweep`` label) — so
    two learning-rate sweeps at different weight decays do not masquerade as
    many two-point weight-decay "sweeps".  (Frames without any ``cd_sweep``
    label skip that requirement.)  Columns: hyper,
    value, loss, status ("completed" / "diverged"), delta (loss minus the
    optimizer's best completed cd loss), final (the slice sits at the accepted
    values of all other hyperparameters), accepted (final slice AND this is
    the accepted value), slice (id within the hyper; earlier-pass slices get
    their own id).  Only phase "cd" rows enter — never the inherited track.
    """
    empty = pd.DataFrame(columns=_SLICE_COLS)
    if df is None or df.empty:
        return empty
    m = ((df["study"] == study) & (df["optimizer"] == opt) & (df["phase"] == "cd")
         & df["status"].isin(["completed", "diverged"]))
    if "budget_multiplier" in df.columns:
        m &= df["budget_multiplier"] == 1.0
    rows = df[m]
    if rows.empty:
        return empty
    recs = []
    for r in rows.to_dict("records"):
        h = _hypers(r, opt)
        ok = r["status"] == "completed" and _num(r.get("final_val_loss")) is not None
        h["loss"] = float(r["final_val_loss"]) if ok else math.nan
        h["status"] = "completed" if ok else "diverged"
        lab = r.get("cd_sweep")
        h["label"] = {"beta1": "momentum"}.get(lab, lab) if isinstance(lab, str) else None
        recs.append(h)
    labelled = any(r["label"] is not None for r in recs)
    comp = [r for r in recs if r["status"] == "completed"]
    best = min(r["loss"] for r in comp) if comp else math.nan
    acc = accepted_config(df, study, opt)
    if acc is None and comp:
        acc = {k: min(comp, key=lambda r: r["loss"])[k] for k in CD_HYPERS}
    out = []
    for h in CD_HYPERS:
        others = [k for k in CD_HYPERS if k != h]
        groups: dict[tuple, dict] = defaultdict(dict)
        swept: set[tuple] = set()
        for r in recs:
            v = r[h]
            if v is None:
                continue
            if r["label"] == h:
                swept.add(tuple(r[k] for k in others))
            cell = groups[tuple(r[k] for k in others)]
            cur = cell.get(v)
            if (cur is None or cur[1] == "diverged"
                    or (r["status"] == "completed" and r["loss"] < cur[0])):
                cell[v] = (r["loss"], r["status"])
        acc_key = None if acc is None else tuple(acc[k] for k in others)
        sid = 0
        for key, cell in groups.items():
            if len(cell) < 2 or (labelled and key not in swept):
                continue
            final = key == acc_key
            for v, (loss, status) in sorted(cell.items()):
                out.append({"hyper": h, "value": v, "loss": loss, "status": status,
                            "delta": loss - best, "final": final,
                            "accepted": bool(final and acc is not None and v == acc[h]),
                            "slice": sid})
            sid += 1
    return pd.DataFrame(out, columns=_SLICE_COLS)


def lr_sweep(df: pd.DataFrame, study: str, opt: str) -> pd.DataFrame:
    """The headline learning-rate sweep (seed 0): lr, loss, status.

    A thin view of ``analysis._lr_sweep_rows(track="headline")`` — the ONE
    implementation of "which rows form the lr sweep": the coordinate-descent
    cells at the ACCEPTED values of every other hyperparameter (so it holds
    the accepted centre, whatever ``cd_sweep`` label dedup left on it), or the
    Stage-A grid for a study without a coordinate descent.  The inherited
    track never enters for a cd-tuned study.  Diverged cells are kept."""
    cols = ["lr", "loss", "status"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    rows = an._lr_sweep_rows(df, study, opt, "headline")
    rows = rows[rows["status"].isin(["completed", "diverged"])]
    recs = {}
    for r in rows.to_dict("records"):
        lr = _num(r.get("lr"))
        if lr is None:
            continue
        ok = r["status"] == "completed" and _num(r.get("final_val_loss")) is not None
        loss = float(r["final_val_loss"]) if ok else math.nan
        cur = recs.get(lr)
        if cur is None or cur[1] == "diverged" or (ok and loss < cur[0]):
            recs[lr] = (loss, "completed" if ok else "diverged")
    return pd.DataFrame([{"lr": k, "loss": v[0], "status": v[1]}
                         for k, v in sorted(recs.items())], columns=cols)


def earlier_pass_divergences(df: pd.DataFrame, study: str, opt: str) -> list[float]:
    """Learning rates at which a coordinate-descent run of (study, optimizer)
    DIVERGED in a pass other than the accepted-context sweep (``lr_sweep``),
    i.e. with other hyperparameters at pre-acceptance values.  They are not on
    the headline curve, but they bound the stable range and must be shown."""
    if df is None or df.empty:
        return []
    r = df[(df["study"] == study) & (df["optimizer"] == opt) & (df["phase"] == "cd")
           & (df["status"] == "diverged")]
    if r.empty:
        return []
    sw = lr_sweep(df, study, opt)
    shown = set(sw[sw["status"] == "diverged"]["lr"])
    return sorted({v for v in (_num(x) for x in r["lr"]) if v is not None} - shown)


def seed_counts(df: pd.DataFrame, study: str) -> dict[str, int]:
    """Completed fresh-seed runs per optimizer of a study (house order)."""
    return {o: int(len(an._fresh(df, study, o))) for o in present_optimizers(df, study)}


def seeds_text(fig_id: str, counts: dict) -> str:
    """The seed count to print in a legend, computed from the data: "3" when
    every group has 3, "2–3" otherwise — and in that case a stdout note names
    the groups with fewer seeds than the others."""
    vals = sorted({int(v) for v in counts.values()})
    if not vals:
        return "0"
    if len(vals) == 1:
        return str(vals[0])
    short = ", ".join(f"{k} ({int(v)})" for k, v in counts.items() if int(v) < vals[-1])
    _note(f"{fig_id}: seed counts differ — fewer than {vals[-1]} seeds for: {short}")
    return f"{vals[0]}–{vals[-1]}"


def spectral_seed_counts(logs: pd.DataFrame | None, df: pd.DataFrame,
                         study: str = "dense") -> dict[str, int]:
    """Fresh seeds with logged final spectral norms, per optimizer."""
    fin = an._fresh_logs(logs, "final", df)
    if fin.empty or "spectral_norms" not in fin.columns:
        return {}
    fin = fin[(fin["study"] == study)
              & fin["spectral_norms"].apply(lambda d: isinstance(d, dict) and len(d) > 0)]
    return {o: int(fin[fin["optimizer"] == o]["seed"].nunique())
            for o in OPT_ORDER if o in set(fin["optimizer"])}


def noise_threshold(df: pd.DataFrame, study: str) -> float:
    """The CD acceptance threshold: 2 x the sample std (ddof=1) of the
    study's pilot final validation losses (studies.pilot_seed_std).  NaN when
    fewer than two completed pilot runs exist; +inf if a pilot run diverged."""
    if df is None or df.empty:
        return math.nan
    p = df[(df["study"] == study) & (df["phase"] == "pilot")]
    if (p["status"] == "diverged").any():
        return math.inf
    v = p[p["status"] == "completed"]["final_val_loss"].dropna().to_numpy(dtype=float)
    if v.size < 2:
        return math.nan
    return 2.0 * float(np.std(v, ddof=1))


def speedup_summary(df: pd.DataFrame, study: str, kind: str = "tokens") -> pd.DataFrame:
    """Speedup over AdamW per optimizer: lo, hi (min-max of the curve-seed x
    run-seed matrix), mid (geometric mean of its finite cells), n_inf,
    n_curve / n_run (the matrix shape: AdamW curve seeds x optimizer seeds),
    extrapolated (some cell lies beyond the largest measured AdamW budget,
    ``analysis.speedup_extrapolation``) and max_budget (that budget, as a
    multiple of 1x).  ``kind``: "tokens" or "flops".  Empty without
    budget-curve runs."""
    cols = ["optimizer", "lo", "hi", "mid", "n_inf", "n_curve", "n_run",
            "extrapolated", "max_budget"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fr = (an.token_speedup_range(study, df) if kind == "tokens"
                  else an.flop_speedup_range(study, df))
            ext = an.speedup_extrapolation(study, df)
        except (ValueError, KeyError, IndexError):
            return pd.DataFrame(columns=cols)
    recs = []
    for opt in [o for o in OPT_ORDER if o in fr.index]:
        shape = np.asarray(fr.loc[opt, "matrix"], dtype=float).shape
        mat = np.asarray(fr.loc[opt, "matrix"], dtype=float).ravel()
        fin = mat[np.isfinite(mat) & (mat > 0)]
        recs.append({
            "optimizer": opt,
            "lo": float(fin.min()) if fin.size else math.inf,
            "hi": float(mat.max()) if mat.size else math.nan,
            "mid": float(np.exp(np.log(fin).mean())) if fin.size else math.inf,
            "n_inf": int(mat.size - fin.size),
            "n_curve": int(shape[0]), "n_run": int(shape[1]),
            "extrapolated": bool(opt in ext.index and ext.loc[opt, "extrapolated"]),
            "max_budget": float(ext.loc[opt, "max_budget"]) if opt in ext.index
            else math.nan,
        })
    return pd.DataFrame(recs, columns=cols)


def budget_fits(df: pd.DataFrame, study: str) -> dict[int, tuple[list, an.FitResult, float]]:
    """Per AdamW curve seed: (points as (budget multiple, loss), the fit in
    budget-multiple units is NOT refitted — the analysis fit in data units is
    returned together with the 1x data size d1)."""
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in an._curve_seeds(study, df):
            try:
                pts = an.curve_points(study, s, df)
                fit = an.fit_budget_curve(pts)
            except ValueError:
                continue
            a = df[(df["study"] == study) & (df["optimizer"] == "adamw")
                   & (df["status"] == "completed") & (df["seed"] == s)]
            d1 = an._unit_size(a)
            out[s] = ([(d / d1, l) for d, l in pts], fit, d1)
    return out


def loss_curves(df: pd.DataFrame, logs: pd.DataFrame | None, study: str,
                x: str = "tokens") -> pd.DataFrame:
    """Validation-loss curves of the completed fresh-seed runs: per optimizer
    and eval index, x (images/tokens, or cumulative FLOPs = the run's measured
    total FLOPs x fraction of the budget done), mean / lo / hi over seeds and
    n_seeds.  The final full-validation loss is appended at the end of the
    budget when the last periodic eval falls short of it."""
    cols = ["optimizer", "k", "x", "mean", "lo", "hi", "n_seeds"]
    ev = an._fresh_logs(logs, "eval", df)
    if ev.empty or "val_loss" not in ev.columns:
        return pd.DataFrame(columns=cols)
    ev = ev[(ev["study"] == study) & ev["val_loss"].notna()
            & ev["images_or_tokens"].notna()]
    fr = an._fresh(df, study)
    if ev.empty or fr.empty:
        return pd.DataFrame(columns=cols)
    recs = []
    for h, g in ev.groupby("hash"):
        res = fr[fr["hash"] == h]
        if res.empty:
            continue
        res = res.iloc[0]
        budget = an._unit_size(fr[fr["hash"] == h])
        g = g.sort_values("images_or_tokens")
        xs = g["images_or_tokens"].to_numpy(dtype=float)
        ys = g["val_loss"].to_numpy(dtype=float)
        if xs[-1] < budget * (1 - 1e-9) and pd.notna(res["final_val_loss"]):
            xs = np.append(xs, budget)
            ys = np.append(ys, float(res["final_val_loss"]))
        scale = 1.0
        if x == "flops":
            if pd.isna(res.get("flops_total")):
                continue
            scale = float(res["flops_total"]) / budget
        for k, (xv, yv) in enumerate(zip(xs, ys)):
            recs.append({"optimizer": res["optimizer"], "seed": res["seed"],
                         "k": k, "x": xv * scale, "y": yv})
    if not recs:
        return pd.DataFrame(columns=cols)
    out = (pd.DataFrame(recs).groupby(["optimizer", "k"])
           .agg(x=("x", "mean"), mean=("y", "mean"), lo=("y", "min"),
                hi=("y", "max"), n_seeds=("seed", "nunique")).reset_index())
    return out[cols]


def optimal_lr_by_width(df: pd.DataFrame) -> pd.DataFrame:
    """F5b data: optimizer, study, width, track ("retuned" = the accepted
    fresh-seed lr; "inherited" = ``analysis.inherited_best_lr``), lr."""
    recs = []
    for study in MLP_STUDIES:
        for opt in OPT_ORDER:
            fr = an._fresh(df, study, opt)
            if fr.empty:
                continue
            w = _num(fr["width"].iloc[0]) if "width" in fr.columns else None
            width = int(w) if w is not None else int(study.replace("mlp", ""))
            recs.append({"optimizer": opt, "study": study, "width": width,
                         "track": "retuned", "lr": an._best_lr(df, study, opt)})
            inh = an.inherited_best_lr(df, study, opt)
            if inh is not None:
                recs.append({"optimizer": opt, "study": study, "width": width,
                             "track": "inherited", "lr": inh})
    return pd.DataFrame(recs, columns=["optimizer", "study", "width", "track", "lr"])


def compute_cost(df: pd.DataFrame, study: str) -> pd.DataFrame:
    """F7 data, from the measured FLOPs of the fresh-seed runs: per optimizer
    the optimizer's FLOPs in units of the model's forward+backward FLOPs
    (mean over seeds), and total = 1 + that."""
    recs = []
    for opt in present_optimizers(df, study):
        fr = an._fresh(df, study, opt)
        r = (fr["flops_optimizer"] / fr["flops_model"]).dropna()
        if r.empty:
            continue
        recs.append({"optimizer": opt, "model": 1.0, "optimizer_cost": float(r.mean()),
                     "total": 1.0 + float(r.mean())})
    return pd.DataFrame(recs, columns=["optimizer", "model", "optimizer_cost", "total"])


def rank_table(df: pd.DataFrame) -> pd.DataFrame:
    """F8 data: per headline study and optimizer the mean fresh-seed loss,
    its rank (1 = best) and ``separated`` — whether the gap to the
    next-better optimizer passes the 3-seed rule (rank 1 has no next-better
    optimizer and is reported separated)."""
    recs = []
    for study in headline_studies(df):
        means = {o: float(an._fresh(df, study, o)["final_val_loss"].mean())
                 for o in present_optimizers(df, study)}
        order = sorted(means, key=means.get)
        for i, opt in enumerate(order):
            sep = True
            if i > 0:
                pair = an._paired_vs(df, study, opt, order[i - 1])
                sep = bool(pair is not None and an.seed_rule(*pair))
            recs.append({"study": study, "optimizer": opt, "mean_loss": means[opt],
                         "rank": i + 1, "separated": sep})
    return pd.DataFrame(recs, columns=["study", "optimizer", "mean_loss", "rank",
                                       "separated"])


def track_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """A2 data (seed 0): per MLP study that has BOTH tuning tracks and per
    optimizer, the re-tuned winner (the cd run at the accepted config) and the
    inherited-settings winner (best stage_a / stage_b / edge run)."""
    recs = []
    for study in MLP_STUDIES:
        for opt in OPT_ORDER:
            inh = df[(df["study"] == study) & (df["optimizer"] == opt)
                     & df["phase"].isin(an.PHASE2_TUNING_PHASES)
                     & (df["status"] == "completed")].dropna(subset=["final_val_loss"])
            acc = accepted_config(df, study, opt)
            cd = df[(df["study"] == study) & (df["optimizer"] == opt)
                    & (df["phase"] == "cd") & (df["status"] == "completed")]
            if inh.empty or acc is None or cd.empty:
                continue
            hit = [r for r in cd.to_dict("records") if _hypers(r, opt) == acc]
            if not hit:
                continue
            b = inh.loc[inh["final_val_loss"].idxmin()]
            recs.append({"study": study, "optimizer": opt,
                         "retuned_loss": float(hit[0]["final_val_loss"]),
                         "retuned_lr": acc["lr"],
                         "inherited_loss": float(b["final_val_loss"]),
                         "inherited_lr": float(b["lr"])})
    return pd.DataFrame(recs, columns=["study", "optimizer", "retuned_loss",
                                       "retuned_lr", "inherited_loss", "inherited_lr"])


def faithful_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """A3(a) data: per study x optimizer with a faithful-settings rerun, the
    faithful loss, the standard fresh-seed loss on the SAME seed, and the
    fresh-seed min / max."""
    recs = []
    f = df[(df["phase"] == "faithful") & (df["status"] == "completed")]
    for r in f.to_dict("records"):
        fr = an._fresh(df, r["study"], r["optimizer"])
        same = fr[fr["seed"] == r["seed"]]
        if fr.empty or same.empty:
            continue
        recs.append({"study": r["study"], "optimizer": r["optimizer"],
                     "seed": int(r["seed"]),
                     "standard": float(same["final_val_loss"].iloc[0]),
                     "faithful": float(r["final_val_loss"]),
                     "lo": float(fr["final_val_loss"].min()),
                     "hi": float(fr["final_val_loss"].max())})
    return pd.DataFrame(recs, columns=["study", "optimizer", "seed", "standard",
                                       "faithful", "lo", "hi"])


def init_ablation_runs(df: pd.DataFrame) -> pd.DataFrame:
    """A3(b) data: study, optimizer, variant ("standard" / "stiefel_init"),
    seed, loss — the init-ablation runs and the matching fresh-seed runs."""
    recs = []
    ia = df[(df["phase"] == "init_ablation") & (df["status"] == "completed")]
    for (study, opt), g in ia.groupby(["study", "optimizer"]):
        for r in g.to_dict("records"):
            recs.append({"study": study, "optimizer": opt, "variant": "stiefel_init",
                         "seed": int(r["seed"]), "loss": float(r["final_val_loss"])})
        for r in an._fresh(df, study, opt).to_dict("records"):
            recs.append({"study": study, "optimizer": opt, "variant": "standard",
                         "seed": int(r["seed"]), "loss": float(r["final_val_loss"])})
    return pd.DataFrame(recs, columns=["study", "optimizer", "variant", "seed", "loss"])


def halflr_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """A3(c) data: per study and seed with a half-LR 2x AdamW run, its loss
    and the constant-LR 2x loss of the same seed."""
    recs = []
    a = df[(df["optimizer"] == "adamw") & (df["status"] == "completed")]
    for r in a[a["phase"] == "budget_halflr"].to_dict("records"):
        c = a[(a["phase"] == "budget_constant") & (a["study"] == r["study"])
              & (a["seed"] == r["seed"])]
        if c.empty:
            continue
        recs.append({"study": r["study"], "seed": int(r["seed"]),
                     "full_lr": float(c["final_val_loss"].iloc[0]),
                     "half_lr": float(r["final_val_loss"])})
    return pd.DataFrame(recs, columns=["study", "seed", "full_lr", "half_lr"])


def lr_stability(df: pd.DataFrame, study: str, opt: str) -> pd.DataFrame:
    """A8 data: every learning rate the optimizer was run at in the study's
    headline tuning phases, with state "ok", "high_loss" (completed but more
    than 0.5 nats above the optimizer's best — the analysis-time divergence
    criterion) or "diverged" (the run stopped), and ``best`` on the accepted
    learning rate."""
    cols = ["lr", "state", "best"]
    phases = an.headline_tuning_phases(df, study)
    r = df[(df["study"] == study) & (df["optimizer"] == opt) & df["phase"].isin(phases)
           & df["status"].isin(["completed", "diverged"])]
    if r.empty:
        return pd.DataFrame(columns=cols)
    comp = r[r["status"] == "completed"]["final_val_loss"].dropna()
    best = float(comp.min()) if not comp.empty else math.inf
    best_lr = an._best_lr(df, study, opt)
    recs = []
    for lr, g in r.groupby(r["lr"].map(_num)):
        losses = g[g["status"] == "completed"]["final_val_loss"].dropna()
        if (g["status"] == "diverged").any():
            state = "diverged"
        elif losses.empty or float(losses.min()) > best + 0.5:
            state = "high_loss"
        else:
            state = "ok"
        recs.append({"lr": float(lr), "state": state,
                     "best": bool(best_lr is not None and _num(best_lr) == lr)})
    return pd.DataFrame(recs, columns=cols).sort_values("lr").reset_index(drop=True)


#: Labels of the Manifold Muon audit runs (docs/manifold_audit.md §3).
AUDIT_LABELS: dict[str, tuple[str, str]] = {
    "baseline_replica": ("manifold", "Frozen-protocol config (cap 5)"),
    "cap0_projection_only": ("manifold", "Cap 0 (Muon step, then project)"),
    "momentum0": ("manifold", "Momentum 0 (reference setting)"),
    "scale0.35": ("manifold", "Forward scale ×0.35"),
    "scale2.8": ("manifold", "Forward scale ×2.8"),
    "scale2.8_lr_x4": ("manifold", "Forward scale ×2.8, classifier+bias lr ×4"),
    "scale8": ("manifold", "Forward scale ×8"),
    "adamw_group_wd0": ("manifold", "Classifier+bias weight decay 0"),
    "adamw_group_lr_x4": ("manifold", "Classifier+bias lr ×4"),
    "adamw_group_lr_x16": ("manifold", "Classifier+bias lr ×16"),
    "adamw_group_lr_x64": ("manifold", "Classifier+bias lr ×64"),
    "adamw_group_lr_x16_wd0": ("manifold", "Classifier+bias lr ×16, weight decay 0"),
    "muon_seed0_reference": ("muon", "Muon, frozen-protocol config"),
    "muon_adamw_group_lr_x16": ("muon", "Muon, classifier+bias lr ×16"),
}


def manifold_audit_rows(root: Path | str) -> pd.DataFrame:
    """A5(a) data from logs/diagnostics/manifold_audit/summary.jsonl: name,
    optimizer, label, loss, status.  Empty if the file is missing."""
    cols = ["name", "optimizer", "label", "loss", "status"]
    p = Path(root) / "logs" / "diagnostics" / "manifold_audit" / "summary.jsonl"
    if not p.exists():
        return pd.DataFrame(columns=cols)
    recs = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        opt, label = AUDIT_LABELS.get(rec["name"], ("manifold", rec["name"]))
        ok = rec.get("status") == "completed" and rec.get("val") is not None
        recs.append({"name": rec["name"], "optimizer": opt, "label": label,
                     "loss": float(rec["val"]) if ok else math.nan,
                     "status": "completed" if ok else "diverged"})
    return pd.DataFrame(recs, columns=cols)


def spectral_by_layer(logs: pd.DataFrame | None, df: pd.DataFrame,
                      study: str = "dense") -> pd.DataFrame:
    """A5(b)/A7 data: final sigma_max per matrix, mean over fresh seeds —
    columns name, short, layer (int or NaN), kind ("q", "k", ..., "embed"),
    optimizer, sigma.  From ``analysis.spectral_norm_means``."""
    cols = ["name", "short", "layer", "kind", "optimizer", "sigma"]
    _, sm = an.spectral_norm_means(logs, study, df)
    if sm.empty:
        return pd.DataFrame(columns=cols)
    recs = []
    for name, row in sm.iterrows():
        short = an.short_layer_name(name)
        layer, kind = math.nan, short
        if short.startswith("L") and "." in short:
            head, kind = short.split(".", 1)
            try:
                layer = int(head[1:])
            except ValueError:
                layer = math.nan
        for opt, v in row.items():
            if pd.notna(v):
                recs.append({"name": name, "short": short, "layer": layer,
                             "kind": kind, "optimizer": opt, "sigma": float(v)})
    return pd.DataFrame(recs, columns=cols)


def attn_logit_bands(logs: pd.DataFrame | None, df: pd.DataFrame,
                     study: str = "dense") -> pd.DataFrame:
    """A6 data: max attention logit of the completed fresh-seed runs, per
    optimizer and token count: mean, lo, hi over seeds, n_seeds."""
    cols = ["optimizer", "x", "mean", "lo", "hi", "n_seeds"]
    ev = an._fresh_logs(logs, "eval", df)
    if ev.empty or "max_attn_logit" not in ev.columns:
        return pd.DataFrame(columns=cols)
    ev = ev[(ev["study"] == study) & ev["max_attn_logit"].notna()]
    if ev.empty:
        return pd.DataFrame(columns=cols)
    out = (ev.groupby(["optimizer", "images_or_tokens"])
           .agg(mean=("max_attn_logit", "mean"), lo=("max_attn_logit", "min"),
                hi=("max_attn_logit", "max"), n_seeds=("seed", "nunique"))
           .reset_index().rename(columns={"images_or_tokens": "x"}))
    return out[cols]


# ===========================================================================
# Drawing helpers

def _study_title(ax, study: str) -> None:
    ax.set_title(STUDY_LABEL.get(study, study))


def _log_lr_axis(ax, lrs, axis: str = "x") -> None:
    """Log axis with decade major ticks and a minor tick on every tested LR."""
    lrs = sorted({float(v) for v in lrs if v and np.isfinite(v) and v > 0})
    if axis == "x":
        ax.set_xscale("log")
        a = ax.xaxis
    else:
        ax.set_yscale("log")
        a = ax.yaxis
    if not lrs:
        return
    lo, hi = lrs[0] / 1.5, lrs[-1] * 1.5
    decades = [10.0 ** k for k in range(math.floor(math.log10(lo)),
                                        math.ceil(math.log10(hi)) + 1)
               if lo <= 10.0 ** k <= hi]
    if len(decades) >= 1:
        a.set_major_locator(FixedLocator(decades))
        a.set_major_formatter(FuncFormatter(
            lambda v, _: f"$10^{{{int(round(math.log10(v)))}}}$" if v > 0 else ""))
    a.set_minor_locator(FixedLocator(lrs))
    a.set_minor_formatter(NullFormatter())
    if axis == "x":
        ax.set_xlim(lo, hi)
    else:
        ax.set_ylim(lo, hi)


def nice_log_ticks(lo: float, hi: float, max_n: int = 4) -> list[float]:
    """Readable tick values for a log axis over [lo, hi]: decades when at
    least three fit, otherwise the coarsest of the 1-3, 1-2-5 or
    1-1.5-2-3-5-7 series that gives three ticks (at most ``max_n``)."""
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0 or hi <= lo:
        return []
    k0, k1 = math.floor(math.log10(lo)) - 1, math.ceil(math.log10(hi)) + 1
    best: list[float] = []
    for series in ((1,), (1, 3), (1, 2, 5), (1, 1.5, 2, 3, 5, 7)):
        t = [m * 10.0 ** k for k in range(k0, k1 + 1) for m in series]
        t = [v for v in t if lo * (1 - 1e-9) <= v <= hi * (1 + 1e-9)]
        if 3 <= len(t) <= max_n:
            return t
        if len(t) > max_n and len(series) == 1:
            step = math.ceil(len(t) / max_n)
            return t[::step]
        if len(best) < 2 and len(t) >= 2 and len(t) <= max_n:
            best = t
        if not best and t:
            best = t[:max_n]
    return best


def _sci_label(v: float) -> str:
    """'10^13' / '3x10^13' in mathtext."""
    if v <= 0:
        return ""
    k = math.floor(math.log10(v) + 1e-9)
    m = v / 10.0 ** k
    if abs(m - 1) < 1e-6:
        return f"$10^{{{k}}}$"
    return f"${m:g}{{\\times}}10^{{{k}}}$"


def _set_log_ticks(ax, lo: float, hi: float, max_n: int = 4, sci: bool = False) -> None:
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ticks = nice_log_ticks(lo, hi, max_n)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(
        (lambda v, _: _sci_label(v)) if sci else (lambda v, _: f"{v:g}")))
    k0, k1 = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
    minor = [m * 10.0 ** k for k in range(k0, k1 + 1) for m in range(1, 10)]
    ax.xaxis.set_minor_locator(FixedLocator([v for v in minor if lo <= v <= hi]))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _key_table(ax, rows: list[tuple[str, str]], header: str,
               xy=(0.985, 0.975), align=(1.0, 1.0)) -> None:
    """A small in-panel table of (optimizer, text) rows, each with the
    optimizer's marker as key; dark text; not part of the layout."""
    kids = [TextArea(header, textprops=dict(color=INK2, fontsize=8))]
    for opt, text in rows:
        da = DrawingArea(10, 8, 0, 0)
        da.add_artist(Line2D([5], [4], marker=OPT_MARKER[opt], color="white",
                             mec=OPT_COLOR[opt], mew=1.3, ms=5.5, linestyle="none"))
        kids.append(HPacker(children=[da, TextArea(text, textprops=dict(
            color=INK, fontsize=8))], align="center", pad=0, sep=3))
    box = VPacker(children=kids, align="left", pad=0, sep=2.5)
    ab = AnnotationBbox(box, xy, xycoords="axes fraction", box_alignment=align,
                        frameon=True, pad=0.35, zorder=6,
                        bboxprops=dict(facecolor="white", edgecolor="none", alpha=0.9))
    ab.set_in_layout(False)
    ax.add_artist(ab)


def _opt_rows(ax, opts: list[str], labels: bool = True) -> dict[str, int]:
    """Categorical optimizer rows on y (first optimizer at the top)."""
    pos = {o: i for i, o in enumerate(opts)}
    ax.set_yticks(range(len(opts)))
    ax.set_yticklabels([OPT_LABEL[o] for o in opts] if labels else [])
    ax.set_ylim(len(opts) - 0.4, -0.6)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    for t in ax.get_yticklabels():
        t.set_color(INK)
    return pos


# ===========================================================================
# Main-text figures

def fig_f1(df, logs, outdir) -> list[Path]:
    """F1 — headline final validation loss, one dot-plot panel per study."""
    studies = headline_studies(df)
    if not studies:
        return _skip("f1_final_loss", "no completed fresh-seed runs")
    opts = [o for o in OPT_ORDER
            if any(not an._fresh(df, s, o).empty for s in studies)]
    with style():
        fig, axes = _subplots(1, len(studies), (FULL_W, 2.45))
        any_div = False
        for j, (ax, study) in enumerate(zip(axes[0], studies)):
            runs = headline_runs(df, study)
            ok = runs[runs["status"] == "completed"]
            pos = _opt_rows(ax, opts, labels=(j == 0))
            lo, hi = float(ok["loss"].min()), float(ok["loss"].max())
            span = max(hi - lo, 1e-9)
            x0, x1 = lo - 0.08 * span, hi + 0.42 * span
            ax.set_xlim(x0, x1)
            a_mean = ok[ok["optimizer"] == "adamw"]["loss"].mean()
            if np.isfinite(a_mean):
                ax.axvline(a_mean, color=INK2, lw=0.8, zorder=1)
            for opt in opts:
                g = ok[ok["optimizer"] == opt]
                y = pos[opt]
                if not g.empty:
                    ax.plot([g["loss"].mean()], [y], marker=OPT_MARKER[opt],
                            color=OPT_COLOR[opt], ms=7.5, linestyle="none", zorder=3)
                    ax.plot(g["loss"], [y + 0.33] * len(g), "o", color=OPT_COLOR[opt],
                            ms=3.4, mec="white", mew=0.5, zorder=4)
                    ax.text(0.985, y, f"{g['loss'].mean():.3f}",
                            transform=ax.get_yaxis_transform(), ha="right",
                            va="center", fontsize=8, color=INK)
                nd = int(((runs["optimizer"] == opt)
                          & (runs["status"] == "diverged")).sum())
                if nd:
                    any_div = True
                    ax.plot([x1 - 0.30 * span] * nd, [y] * nd, "x", color=INK,
                            mec=INK, ms=6, mew=1.4, zorder=5)
            ax.xaxis.set_major_locator(MaxNLocator(3, min_n_ticks=3))
            _study_title(ax, study)
        fig.supxlabel("Final validation loss (lower is better)", fontsize=9)
        n_txt = seeds_text("f1_final_loss", {f"{s}/{o}": n for s in studies
                                             for o, n in seed_counts(df, s).items()})
        handles = [
            Line2D([], [], marker="o", color=PALE, ms=7.5, linestyle="none",
                   label=f"mean of {n_txt} seeds (value at right)"),
            Line2D([], [], marker="o", color=PALE, ms=3.4, mec="white", mew=0.5,
                   linestyle="none", label="one seed (row below the mean)"),
            Line2D([], [], color=INK2, lw=0.8, label="AdamW mean"),
        ]
        if any_div:
            handles.append(Line2D([], [], marker="x", color=INK, mec=INK, ms=6, mew=1.4,
                                  linestyle="none", label="diverged run"))
        _legend(fig, handles, ncol=len(handles))
        return _save(fig, outdir, "f1_final_loss")


def fig_f2(df, logs, outdir) -> list[Path]:
    """F2 — token/image and FLOP speedup over AdamW (dot + min-max whisker)."""
    studies = [s for s in headline_studies(df)
               if not speedup_summary(df, s, "tokens").empty]
    if not studies:
        return _skip("f2_speedup", "no AdamW budget-curve runs")
    kinds = (("tokens", "Per token / image"), ("flops", "Per FLOP"))
    data = {(k, s): speedup_summary(df, s, k) for k, _ in kinds for s in studies}
    opts = [o for o in OPT_ORDER if any(o in set(d["optimizer"]) for d in data.values())]
    with style():
        fig, axes = _subplots(2, len(studies), (FULL_W, 3.9), sharey=True)
        any_inf = False
        extrap_budgets: set[float] = set()
        for i, (kind, rowlab) in enumerate(kinds):
            vals = []
            for s in studies:
                d = data[(kind, s)]
                vals += [v for v in list(d["lo"]) + list(d["hi"]) + list(d["mid"])
                         if np.isfinite(v) and v > 0]
            vmin, vmax = min(vals + [1.0]), max(vals + [1.0])
            ratio = vmax / vmin
            x0, x1 = vmin / ratio ** 0.06, vmax * ratio ** 0.34
            for j, study in enumerate(studies):
                ax = axes[i][j]
                d = data[(kind, study)].set_index("optimizer")
                pos = _opt_rows(ax, opts, labels=(j == 0))
                _set_log_ticks(ax, x0, x1, max_n=5)
                ax.axvspan(x0, 1.0, color=SHADE, zorder=0, lw=0)
                ax.axvline(1.0, color=INK2, lw=0.8, zorder=1)
                for opt in opts:
                    if opt not in d.index:
                        continue
                    r, y, c = d.loc[opt], pos[opt], OPT_COLOR[opt]
                    hi = r["hi"]
                    if not np.isfinite(r["mid"]):
                        any_inf = True
                        ax.annotate("", xy=(x1, y), xytext=(1.0, y),
                                    arrowprops=dict(arrowstyle="->", color=c, lw=1.4))
                        ax.text(1.0, y - 0.28, "∞", ha="left", va="bottom",
                                fontsize=8, color=INK)
                        continue
                    end = hi
                    if not np.isfinite(hi):          # some cells unbounded
                        any_inf = True
                        end = x1
                        ax.annotate("", xy=(x1, y), xytext=(r["mid"], y),
                                    arrowprops=dict(arrowstyle="->", color=c, lw=1.4))
                        ax.plot([r["lo"], r["mid"]], [y, y], color=c, lw=1.6,
                                solid_capstyle="butt", zorder=2)
                    else:
                        ax.plot([r["lo"], hi], [y, y], color=c, lw=1.6,
                                solid_capstyle="butt", zorder=2)
                        ax.plot([r["lo"], hi], [y, y], "|", color=c, mec=c, ms=6, mew=1.4,
                                zorder=2)
                    if r["extrapolated"]:
                        extrap_budgets.add(float(r["max_budget"]))
                        ax.plot([r["mid"]], [y], marker=OPT_MARKER[opt], mfc="white",
                                mec=c, mew=1.6, ms=6.5, linestyle="none", zorder=3)
                    else:
                        ax.plot([r["mid"]], [y], marker=OPT_MARKER[opt], color=c,
                                linestyle="none", zorder=3)
                    if np.isfinite(hi):
                        ax.annotate(fmt_mult(r["mid"]), (end, y), xytext=(5, 0),
                                    textcoords="offset points", va="center",
                                    ha="left", fontsize=8, color=INK)
                    else:
                        ax.text(r["mid"], y - 0.28, f"{fmt_mult(r['mid'])} … ∞",
                                ha="left", va="bottom", fontsize=8, color=INK)
                if i == 0:
                    _study_title(ax, study)
                if j == 0:
                    ax.set_ylabel(rowlab, fontsize=9, fontweight="bold")
        fig.supxlabel("Speedup over AdamW (×, log scale): AdamW data or FLOPs to reach "
                      "the same loss ÷ the optimizer's own", fontsize=9)
        tok = {s: data[("tokens", s)] for s in studies}
        n_curve = seeds_text("f2_speedup (AdamW curve seeds)",
                             {s: int(d["n_curve"].max()) for s, d in tok.items()})
        n_run = seeds_text("f2_speedup (run seeds)",
                           {f"{s}/{o}": int(n) for s, d in tok.items()
                            for o, n in zip(d["optimizer"], d["n_run"])})
        handles = [
            Line2D([], [], marker="o", color=PALE, lw=1.6, ms=6.5,
                   label=f"geometric mean, min–max over {n_curve} curve seeds × "
                         f"{n_run} run seeds"),
            Line2D([], [], color=INK2, lw=0.8, label="AdamW (= 1)"),
            Patch(facecolor=SHADE, edgecolor=GRID, label="slower than AdamW"),
        ]
        if extrap_budgets:
            b = sorted(extrap_budgets)
            b_s = f"{b[0]:g}×" if len(b) == 1 else f"{b[0]:g}×–{b[-1]:g}×"
            handles.append(Line2D([], [], marker="o", mfc="white", mec=PALE, mew=1.6,
                                  ms=6.5, linestyle="none",
                                  label=f"extrapolated beyond the largest AdamW budget "
                                        f"({b_s})"))
        if any_inf:
            handles.append(Line2D([], [], color=PALE, lw=1.4, marker=">", ms=4,
                                  label="∞: below AdamW's fitted asymptote"))
        _legend(fig, handles, ncol=2 if len(handles) > 3 else len(handles))
        return _save(fig, outdir, "f2_speedup")


def fig_f3(df, logs, outdir) -> list[Path]:
    """F3 — the AdamW loss-vs-data curve and how a speedup is read off it."""
    studies = [s for s in headline_studies(df) if budget_fits(df, s)]
    if not studies:
        return _skip("f3_budget_curve", "no AdamW budget-curve runs")
    ncols = 2 if len(studies) > 1 else 1
    nrows = math.ceil(len(studies) / ncols)
    with style():
        fig, axes = _subplots(nrows, ncols, (FULL_W, 2.75 * nrows + 0.45))
        flat = list(axes.ravel())
        pending = []
        extrap = False
        for ax, study in zip(flat, studies):
            fits = budget_fits(df, study)
            sp = speedup_summary(df, study, "tokens").set_index("optimizer")
            runs = headline_runs(df, study)
            runs = runs[runs["status"] == "completed"]
            means = runs.groupby("optimizer")["loss"].mean()
            mids = [float(sp.loc[o, "mid"]) for o in sp.index
                    if o != "adamw" and np.isfinite(sp.loc[o, "mid"])]
            xs_pts = [x for pts, _, _ in fits.values() for x, _ in pts]
            xmax_data = max(xs_pts)
            x_lo = min(xs_pts) / 1.2
            x_hi = max([xmax_data] + mids) * 1.75
            for s, (pts, fit, d1) in sorted(fits.items()):
                gx = np.geomspace(x_lo, xmax_data, 200)
                ax.plot(gx, [fit(v * d1) for v in gx], color=OPT_COLOR["adamw"],
                        lw=1.0, ls=(0, (4, 2)), alpha=0.8, zorder=2)
                if x_hi > xmax_data:
                    extrap = extrap or any(m > xmax_data for m in mids)
                    gx2 = np.geomspace(xmax_data, x_hi, 60)
                    ax.plot(gx2, [fit(v * d1) for v in gx2], color=OPT_COLOR["adamw"],
                            lw=1.0, ls=(0, (1, 2)), alpha=0.8, zorder=2)
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "o", color=PALE,
                        ms=4.2, mec="white", mew=0.6, zorder=3)
            labels = []
            for opt in [o for o in OPT_ORDER if o in means.index]:
                L, c = float(means[opt]), OPT_COLOR[opt]
                if opt != "adamw" and opt in sp.index and np.isfinite(sp.loc[opt, "mid"]):
                    mid = float(sp.loc[opt, "mid"])
                    ax.plot([1.0, mid], [L, L], color=c, lw=1.0, zorder=3)
                    ax.plot([mid], [L], marker=OPT_MARKER[opt], color="white", mec=c,
                            mew=1.3, ms=5.5, linestyle="none", zorder=5)
                    labels.append((opt, f"{OPT_LABEL[opt]}  {fmt_mult(mid)}"))
                ax.plot([1.0], [L], marker=OPT_MARKER[opt], color=c, linestyle="none",
                        ms=7, zorder=4 if opt != "adamw" else 3.5)
            ax.set_xscale("log", base=2)
            ax.set_xlim(x_lo, x_hi)
            ticks = [t for t in (0.125, 0.25, 0.5, 1, 2, 4, 8) if x_lo <= t <= x_hi]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
            ax.xaxis.set_minor_locator(FixedLocator([]))
            ys = [l for pts, _, _ in fits.values() for _, l in pts] + list(means)
            pad = 0.05 * (max(ys) - min(ys))
            ax.set_ylim(min(ys) - 1.6 * pad, max(ys) + pad)
            _study_title(ax, study)
            pending.append((ax, labels))
        for ax in flat[len(studies):]:
            ax.set_visible(False)
        fig.supxlabel("Training data, as a multiple of the 1× budget (log scale)",
                      fontsize=9)
        fig.supylabel("Final validation loss (lower is better)", fontsize=9)
        handles = [_opt_handle(o, line=False) for o in OPT_ORDER
                   if any(o in present_optimizers(df, s) for s in studies)]
        for h in handles:
            h.set_label(h.get_label() + " at 1×")
        n_txt = seeds_text("f3_budget_curve (AdamW curve seeds)",
                           {s: len(budget_fits(df, s)) for s in studies})
        handles += [
            Line2D([], [], marker="o", color=PALE, ms=4.2, mec="white", mew=0.6,
                   linestyle="none", label=f"AdamW run ({n_txt} seeds per budget)"),
            Line2D([], [], color=OPT_COLOR["adamw"], lw=1.0, ls=(0, (4, 2)),
                   label="fit L(D) = αD$^{-β}$ + γ, one per AdamW seed"),
        ]
        if extrap:
            handles.append(Line2D([], [], color=OPT_COLOR["adamw"], lw=1.0,
                                  ls=(0, (1, 2)), label="fit extrapolated beyond 2×"))
        handles.append(Line2D([], [], color=PALE, lw=1.0, marker="s", mfc="white",
                              mec=PALE, mew=1.3, ms=5.5,
                              label="AdamW data reaching the same loss (hollow marker)"))
        _legend(fig, handles, ncol=3)
        for ax, labels in pending:
            if labels:
                _key_table(ax, labels, "AdamW data for the same loss")
        return _save(fig, outdir, "f3_budget_curve")


def _curve_figure(df, logs, outdir, stem: str, xkind: str) -> list[Path]:
    studies = [s for s in headline_studies(df)
               if not loss_curves(df, logs, s, xkind).empty]
    if not studies:
        return _skip(stem, "no fresh-seed eval logs")
    with style():
        fig, axes = _subplots(2, len(studies), (FULL_W, 4.3))
        used, n_by = [], {}
        for j, study in enumerate(studies):
            cur = loss_curves(df, logs, study, xkind)
            for o, g in cur.groupby("optimizer"):
                n_by[f"{study}/{o}"] = int(g["n_seeds"].min())
            unit = "images" if study.startswith("mlp") else "tokens"
            opts = [o for o in OPT_ORDER if o in set(cur["optimizer"])]
            used += [o for o in opts if o not in used]
            tok = loss_curves(df, logs, study, "tokens")
            budget = float(tok["x"].max())
            div = 1.0 if xkind == "flops" else 1e6
            # informative ranges
            tail = {}       # per optimizer: rows in the last 25% of ITS run
            for o in opts:
                g = cur[cur["optimizer"] == o].sort_values("k")
                t = tok[tok["optimizer"] == o].sort_values("k")
                frac = t["x"].to_numpy() / budget
                tail[o] = g[frac >= 0.75 - 1e-9]
            finals = [float(cur[cur["optimizer"] == o].sort_values("k")["mean"].iloc[-1])
                      for o in opts]
            # top row: clip the first evals — show from 10% of the run on
            top_hi = []
            for o in opts:
                g = cur[cur["optimizer"] == o].sort_values("k")
                t = tok[tok["optimizer"] == o].sort_values("k")
                frac = t["x"].to_numpy() / budget
                sel = g[frac >= 0.10]
                if not sel.empty:
                    top_hi.append(float(sel["hi"].max()))
            y_lo = min(float(cur["lo"].min()), min(finals))
            y_top = max(top_hi) if top_hi else float(cur["hi"].max())
            for i in (0, 1):
                ax = axes[i][j]
                for o in opts:
                    g = cur[cur["optimizer"] == o].sort_values("k")
                    ax.fill_between(g["x"] / div, g["lo"], g["hi"], color=OPT_COLOR[o],
                                    alpha=0.18, lw=0, zorder=2)
                    ax.plot(g["x"] / div, g["mean"], color=OPT_COLOR[o], lw=1.8,
                            zorder=3)
                    last = g.iloc[-1]
                    ax.plot([last["x"] / div], [last["mean"]], marker=OPT_MARKER[o],
                            color=OPT_COLOR[o], linestyle="none", ms=5.5, mew=0.8,
                            zorder=4, clip_on=False)
            ax0, ax1 = axes[0][j], axes[1][j]
            pad = 0.04 * (y_top - y_lo)
            ax0.set_ylim(y_lo - pad, y_top + pad)
            t_lo = min(float(tail[o]["lo"].min()) for o in opts)
            t_hi = max(float(tail[o]["hi"].max()) for o in opts)
            pad1 = 0.06 * (t_hi - t_lo)
            ax1.set_ylim(t_lo - pad1, t_hi + pad1)
            if xkind == "tokens":
                ax0.set_xlim(0, budget / div * 1.03)
                ax1.set_xlim(0.75 * budget / div, budget / div * 1.012)
                ax0.xaxis.set_major_locator(MaxNLocator(4))
                ax1.xaxis.set_major_locator(MaxNLocator(3))
                ax1.set_xlabel(f"{unit.capitalize()} seen (millions)")
            else:
                xs = cur["x"].to_numpy(dtype=float)
                in0 = cur[cur["mean"] <= y_top + pad]["x"]
                in1 = cur[cur["mean"] <= t_hi + pad1]["x"]
                _set_log_ticks(ax0, float(in0.min()) / 1.3, xs.max() * 1.3, max_n=3,
                               sci=True)
                _set_log_ticks(ax1, float(in1.min()) / 1.3, xs.max() * 1.3, max_n=3,
                               sci=True)
            _study_title(ax0, study)
            if j == 0:
                ax0.set_ylabel("Validation loss\n(lower is better)")
                ax1.set_ylabel("Zoom: last 25% of each run" if xkind == "tokens"
                               else "Zoom on the final losses")
        if xkind == "flops":
            fig.supxlabel("Cumulative training FLOPs (log scale)", fontsize=9)
        handles = [_opt_handle(o) for o in used]
        handles.append(Patch(facecolor=PALE, alpha=0.35, lw=0,
                             label=f"min–max over {seeds_text(stem, n_by)} seeds "
                                   f"(line: mean)"))
        _legend(fig, handles, ncol=len(handles) if len(handles) <= 6
                else math.ceil(len(handles) / 2))
        return _save(fig, outdir, stem)


def fig_f4(df, logs, outdir) -> list[Path]:
    """F4 — validation loss vs images/tokens (mean line, min-max band)."""
    return _curve_figure(df, logs, outdir, "f4_loss_curves_tokens", "tokens")


def fig_f4b(df, logs, outdir) -> list[Path]:
    """F4b — validation loss vs cumulative FLOPs (log x)."""
    return _curve_figure(df, logs, outdir, "f4b_loss_curves_flops", "flops")


def _mlp_widths(df) -> list[tuple[str, int]]:
    out = []
    for s in MLP_STUDIES:
        if (df["study"] == s).any():
            out.append((s, int(s.replace("mlp", ""))))
    return out


def fig_f5(df, logs, outdir) -> list[Path]:
    """F5 — loss vs learning rate per MLP width, one panel per optimizer (H4)."""
    widths = [(s, w) for s, w in _mlp_widths(df)
              if any(not lr_sweep(df, s, o).empty for o in OPT_ORDER)]
    if not widths:
        return _skip("f5_width_transfer", "no MLP learning-rate sweeps")
    opts = [o for o in OPT_ORDER if any(not lr_sweep(df, s, o).empty for s, _ in widths)]
    clip = 0.15
    # up to five optimizers: one block (a row of loss panels over a row of
    # basin panels); more: two blocks, so a panel stays wide enough to read
    per_row = len(opts) if len(opts) <= 5 else math.ceil(len(opts) / 2)
    blocks = math.ceil(len(opts) / per_row)
    with style():
        if blocks == 1:
            fig, axes = _subplots(2, per_row, (FULL_W, 4.0), sharex="col")
        else:
            fig, axes = _subplots(2 * blocks, per_row, (FULL_W, 3.6 * blocks + 0.5))
        pairs = [(axes[2 * (j // per_row)][j % per_row],
                  axes[2 * (j // per_row) + 1][j % per_row]) for j in range(len(opts))]
        if blocks > 1:
            for a0, a1 in pairs:
                a0.sharex(a1)
                a0.tick_params(labelbottom=False)
            for ax in axes.ravel():     # an odd count leaves one pair empty
                if not any(ax is a for pr in pairs for a in pr):
                    ax.set_visible(False)
        any_div = any_raw = any_early = False
        for j, opt in enumerate(opts):
            a0, a1 = pairs[j]
            all_lrs, sel, tops = [], {}, []
            early: set[float] = set()
            for study, w in widths:
                sw = lr_sweep(df, study, opt)
                if sw.empty:
                    continue
                c = WIDTH_COLOR.get(w, INK2)
                ok = sw[sw["status"] == "completed"]
                all_lrs += list(sw["lr"])
                early |= set(earlier_pass_divergences(df, study, opt))
                if ok.empty:
                    continue
                lmin = float(ok["loss"].min())
                tops.append(float(ok["loss"].max()))
                a0.plot(ok["lr"], ok["loss"], "-o", color=c, ms=4.2, mew=0.6, zorder=3)
                a1.plot(ok["lr"], ok["loss"] - lmin, "-o", color=c, ms=4.2, mew=0.6,
                        zorder=3)
                best_lr = an._best_lr(df, study, opt)
                raw_lr = float(ok.loc[ok["loss"].idxmin(), "lr"])
                hit = ok[np.isclose(ok["lr"], best_lr)] if best_lr is not None else ok.iloc[0:0]
                if not hit.empty:
                    sel[w] = float(best_lr)
                    for ax, y in ((a0, float(hit["loss"].iloc[0])),
                                  (a1, float(hit["loss"].iloc[0]) - lmin)):
                        ax.plot([best_lr], [y], "o", color=c, ms=7.5, mec=INK, mew=1.1,
                                zorder=5)
                    if not math.isclose(raw_lr, best_lr):
                        any_raw = True
                        for ax, y in ((a0, lmin), (a1, 0.0)):
                            ax.plot([raw_lr], [y], "o", mfc="none", mec=INK, mew=1.0,
                                    ms=8.5, zorder=5)
                for lr in sw[sw["status"] == "diverged"]["lr"]:
                    any_div = True
                    for ax in (a0, a1):
                        ax.plot([lr], [0.965], "x", color=INK, mec=INK, ms=6, mew=1.5,
                                transform=ax.get_xaxis_transform(), zorder=6,
                                clip_on=False)
            for lr in sorted(early):
                any_early = True
                for ax in (a0, a1):
                    ax.plot([lr], [0.965], "x", color=INK2, mec=INK2, ms=5, mew=1.0,
                            transform=ax.get_xaxis_transform(), zorder=6,
                            clip_on=False)
            _log_lr_axis(a1, all_lrs + sorted(early))
            a0.set_title(OPT_LABEL[opt])
            a1.set_ylim(-0.03, clip)
            a0.yaxis.set_major_locator(MaxNLocator(4))
            if tops:
                y0, y1 = a0.get_ylim()
                a0.set_ylim(y0, y1 + 0.08 * (y1 - y0))
            ws = sorted(sel)
            if len(ws) >= 2:
                lo_w, hi_w = ws[0], ws[-1]
                txt = fmt_shift(sel[hi_w] / sel[lo_w])
                if not math.isclose(sel[hi_w], sel[lo_w]):
                    a1.annotate("", xy=(sel[hi_w], -0.017), xytext=(sel[lo_w], -0.017),
                                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0,
                                                shrinkA=0, shrinkB=0,
                                                mutation_scale=7))
                a1.set_title(f"{lo_w}→{hi_w}: {txt}", fontsize=8, pad=3)
            if j % per_row > 0:
                a1.tick_params(labelleft=False)
        for _, a1 in pairs[1:]:
            a1.sharey(pairs[0][1])
        pairs[0][1].set_ylim(-0.03, clip)
        for a0, a1 in pairs[::per_row]:
            a0.set_ylabel("Final validation loss\n(lower is better)")
            a1.set_ylabel("Loss − curve minimum")
        fig.supxlabel("Learning rate (log scale; ticks mark the tested values, "
                      "a factor 2 apart)", fontsize=9)
        handles = [Line2D([], [], color=WIDTH_COLOR.get(w, INK2), marker="o", ms=4.2,
                          mew=0.6, label=STUDY_LABEL[s]) for s, w in widths]
        handles.append(Line2D([], [], marker="o", color=PALE, mec=INK, mew=1.1, ms=7.5,
                              linestyle="none", label="selected LR"))
        if any_raw:
            handles.append(Line2D([], [], marker="o", mfc="none", mec=INK, mew=1.0,
                                  ms=8.5, linestyle="none",
                                  label="seed-0 minimum (within noise of selected)"))
        if any_div:
            handles.append(Line2D([], [], marker="x", color=INK, mec=INK, ms=6, mew=1.5,
                                  linestyle="none", label="diverged"))
        if any_early:
            handles.append(Line2D([], [], marker="x", color=INK2, mec=INK2, ms=5,
                                  mew=1.0, linestyle="none",
                                  label="diverged in an earlier tuning pass"))
        _legend(fig, handles, ncol=3 if len(handles) > 5 else len(handles),
                columnspacing=1.1)
        return _save(fig, outdir, "f5_width_transfer")


def build_f5b(df):
    """The F5b figure object (None if there is nothing to draw)."""
    tab = optimal_lr_by_width(df)
    tab = tab.dropna(subset=["lr"])
    if tab.empty:
        return None
    opts = [o for o in OPT_ORDER if o in set(tab["optimizer"])]
    widths = sorted(set(tab["width"]))
    with style():
        fig, axes = _subplots(1, len(opts), (FULL_W, 2.45), sharey=True)
        for j, (ax, opt) in enumerate(zip(axes[0], opts)):
            c, mk = OPT_COLOR[opt], OPT_MARKER[opt]
            t = tab[tab["optimizer"] == opt]
            inh = t[t["track"] == "inherited"].sort_values("width")
            ret = t[t["track"] == "retuned"].sort_values("width")
            if not inh.empty:
                ax.plot(inh["width"], inh["lr"], ls=(0, (3, 2)), color=c, lw=1.4,
                        marker=mk, mfc="white", mec=c, mew=1.4, ms=11, zorder=3,
                        gid="inherited")
            ax.plot(ret["width"], ret["lr"], "-", color=c, marker=mk, ms=6.5, zorder=4,
                    gid="retuned")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log", base=2)
            ax.set_xlim(min(widths) / 1.35, max(widths) * 1.35)
            ax.xaxis.set_major_locator(FixedLocator(widths))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_locator(FixedLocator([]))
            ax.set_title(OPT_LABEL[opt])
        lrs = sorted(set(tab["lr"].map(_num)))
        ax0 = axes[0][0]
        ax0.set_ylim(min(lrs) / 1.5, max(lrs) * 1.5)
        ax0.yaxis.set_major_locator(FixedLocator(lrs))
        ax0.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_num(v)))
        ax0.yaxis.set_minor_locator(FixedLocator([]))
        ax0.set_ylabel("Selected learning rate\n(log scale)")
        fig.supxlabel("MLP width", fontsize=9)
        handles = [
            Line2D([], [], color=PALE, marker="o", ms=6.5, lw=1.8,
                   label="re-tuned at this width (coordinate descent)"),
            Line2D([], [], color=PALE, marker="o", mfc="white", mec=PALE, mew=1.4,
                   ms=11, lw=1.4, ls=(0, (3, 2)),
                   label="inherited MLP-128 settings + LR × WD grid (seed 0)"),
        ]
        _legend(fig, handles, ncol=2)
    return fig


def fig_f5b(df, logs, outdir) -> list[Path]:
    """F5b — selected learning rate vs width, re-tuned vs inherited track."""
    fig = build_f5b(df)
    if fig is None:
        return _skip("f5b_optimal_lr_vs_width", "no MLP fresh-seed runs")
    with style():
        return _save(fig, outdir, "f5b_optimal_lr_vs_width")


def fig_f6(df, logs, outdir, study: str) -> list[Path]:
    """F6 — coordinate-descent sensitivity grid of one study."""
    stem = f"f6_hparam_sensitivity_{study}"
    opts = [o for o in OPT_ORDER if not cd_slices(df, study, o).empty]
    if not opts:
        return _skip(stem, "no coordinate-descent runs")
    thr = noise_threshold(df, study)
    hypers = f6_columns(opts)
    narrow = len(hypers) > 5        # the extra alpha column: tighter text
    titles = CD_HYPER_LABEL_NARROW if narrow else CD_HYPER_LABEL
    with style():
        fig, axes = _subplots(len(opts), len(hypers),
                              (FULL_W, 1.22 * len(opts) + 1.15))
        flags = dict(earlier=False, over=False, div=False, star=False)
        for i, opt in enumerate(opts):
            sl = cd_slices(df, study, opt)
            c = OPT_COLOR[opt]
            ok = sl[sl["status"] == "completed"]
            base = ok[ok["hyper"] != "lr"]["delta"]
            near = ok[(ok["hyper"] == "lr") & ok["final"]]["delta"].nsmallest(3)
            cands = [float(base.max())] if not base.empty else []
            if not near.empty:
                cands.append(float(near.max()))
            ytop = max(cands + ([2.6 * thr] if np.isfinite(thr) else []) + [1e-6]) * 1.22
            for j, hyper in enumerate(hypers):
                ax = axes[i][j]
                h = sl[sl["hyper"] == hyper]
                if i == 0:
                    ax.set_title(titles[hyper])
                if narrow:
                    ax.tick_params(axis="x", labelsize=7)
                if j == 0:
                    ax.set_ylabel(OPT_LABEL[opt], fontsize=9, fontweight="bold")
                ax.set_ylim(-0.07 * ytop, ytop)
                ax.yaxis.set_major_locator(MaxNLocator(3, min_n_ticks=3))
                if j > 0:
                    ax.tick_params(labelleft=False)
                if h.empty:
                    ax.grid(False)
                    ax.set_xticks([])
                    ax.tick_params(left=False)
                    for sp in ("left", "bottom"):
                        ax.spines[sp].set_color(GRID)
                    ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center",
                            va="center", fontsize=8, color=PALE)
                    continue
                if np.isfinite(thr):
                    ax.axhspan(0, min(thr, ytop), color=NEUTRAL, alpha=0.55, lw=0,
                               zorder=0)
                values = sorted(set(h["value"]))
                if hyper == "lr":
                    xmap = {v: v for v in values}
                    _log_lr_axis(ax, values)
                elif hyper == "alpha":
                    # a ratio on a ×~3 grid: log x, every tested value ticked,
                    # every other one labelled when all would collide
                    xmap = {v: v for v in values}
                    ax.set_xscale("log")
                    ax.set_xlim(min(values) / 1.7, max(values) * 1.7)
                    major = values[::2] if len(values) > 4 else values
                    ax.xaxis.set_major_locator(FixedLocator(major))
                    ax.xaxis.set_major_formatter(FuncFormatter(
                        lambda v, _: fmt_cat(v)))
                    ax.xaxis.set_minor_locator(FixedLocator(
                        [v for v in values if v not in major]))
                    ax.xaxis.set_minor_formatter(NullFormatter())
                    ax.grid(axis="x", visible=False)
                else:
                    xmap = {v: k for k, v in enumerate(values)}
                    ax.set_xlim(-0.5, len(values) - 0.5)
                    ax.set_xticks(range(len(values)))
                    ax.set_xticklabels([fmt_cat(v) for v in values])
                    ax.grid(axis="x", visible=False)
                slices = sorted(h["slice"].unique(),
                                key=lambda s: bool(h[h["slice"] == s]["final"].iloc[0]))
                for sid in slices:
                    g = h[h["slice"] == sid].sort_values("value")
                    final = bool(g["final"].iloc[0])
                    flags["earlier"] |= not final
                    gc = g[g["status"] == "completed"]
                    xs = [xmap[v] for v in gc["value"]]
                    inr = gc["delta"] <= ytop
                    kw = (dict(color=c, lw=1.6, zorder=3) if final
                          else dict(color=c, lw=1.0, alpha=0.5, zorder=2))
                    ax.plot(xs, gc["delta"], "-", **kw)
                    xi = [x for x, m in zip(xs, inr) if m]
                    if final:
                        ax.plot(xi, gc["delta"][inr], linestyle="none",
                                marker=OPT_MARKER[opt], color=c, ms=5, mew=0.7, zorder=4)
                    else:
                        ax.plot(xi, gc["delta"][inr], linestyle="none",
                                marker=OPT_MARKER[opt], mfc="white", mec=c, mew=1.0,
                                ms=4.2, alpha=0.75, zorder=3)
                    xo = [x for x, m in zip(xs, inr) if not m]
                    if xo:
                        flags["over"] = True
                        ax.plot(xo, [0.955] * len(xo), "^", color=c if final else "white",
                                mec="white" if final else c, mew=0.6 if final else 0.9,
                                ms=4.2, alpha=1.0 if final else 0.75,
                                transform=ax.get_xaxis_transform(), zorder=5)
                    xd = [xmap[v] for v in g[g["status"] == "diverged"]["value"]]
                    if xd:
                        flags["div"] = True
                        ax.plot(xd, [0.955] * len(xd), "x", color=INK, mec=INK, ms=5.5, mew=1.4,
                                transform=ax.get_xaxis_transform(), zorder=6)
                    acc = gc[gc["accepted"]]
                    if final and not acc.empty:
                        flags["star"] = True
                        ax.plot([xmap[acc["value"].iloc[0]]],
                                [min(float(acc["delta"].iloc[0]), ytop)], "*", color=c,
                                mec=INK, mew=0.7, ms=11, zorder=7)
        fig.supylabel("Final validation loss − the optimizer's best "
                      "(lower is better; axis clipped per row)", fontsize=9)
        handles = [Line2D([], [], color=PALE, marker="o", ms=5, mew=0.7, lw=1.6,
                          label="sweep around the accepted configuration")]
        if flags["earlier"]:
            handles.append(Line2D([], [], color=PALE, marker="o", mfc="white", mec=PALE,
                                  mew=1.0, ms=4.2, lw=1.0, alpha=0.75,
                                  label="earlier pass (before a change was accepted)"))
        if flags["star"]:
            handles.append(Line2D([], [], marker="*", color=PALE, mec=INK, mew=0.7,
                                  ms=11, linestyle="none", label="accepted value"))
        if np.isfinite(thr):
            handles.append(Patch(facecolor=NEUTRAL, alpha=0.55, lw=0,
                                 label=f"noise: 2 × pilot seed std = {thr:.3f} "
                                       f"(acceptance threshold)"))
        if flags["over"]:
            handles.append(Line2D([], [], marker="^", color=PALE, ms=5, mew=0.7,
                                  linestyle="none", label="above the axis limit"))
        if flags["div"]:
            handles.append(Line2D([], [], marker="x", color=INK, mec=INK, ms=5.5, mew=1.4,
                                  linestyle="none", label="diverged"))
        _legend(fig, handles, ncol=2,
                title=f"{STUDY_LABEL.get(study, study)} — coordinate descent, seed 0",
                title_fontsize=10)
        return _save(fig, outdir, stem)


def fig_f7(df, logs, outdir) -> list[Path]:
    """F7 — where the FLOPs go: model vs optimizer, per study."""
    studies = [s for s in headline_studies(df) if not compute_cost(df, s).empty]
    if not studies:
        return _skip("f7_compute_cost", "no FLOP measurements")
    opts = [o for o in OPT_ORDER if any(o in set(compute_cost(df, s)["optimizer"])
                                        for s in studies)]
    with style():
        fig, axes = _subplots(1, len(studies), (FULL_W, 2.5))
        any_clip = False
        for j, (ax, study) in enumerate(zip(axes[0], studies)):
            d = compute_cost(df, study).set_index("optimizer")
            pos = _opt_rows(ax, opts, labels=(j == 0))
            tot = d["total"].sort_values()
            second = float(tot.iloc[-2]) if len(tot) > 1 else float(tot.iloc[-1])
            clip_at = None
            if float(tot.iloc[-1]) > 2.5 * second:
                clip_at = 1.22 * second
            xmax = (clip_at if clip_at else float(tot.iloc[-1])) * 1.42
            ax.set_xlim(0, xmax)
            for opt in opts:
                if opt not in d.index:
                    continue
                y, r = pos[opt], d.loc[opt]
                end = r["total"]
                clipped = clip_at is not None and end > clip_at
                if clipped:
                    end = clip_at
                    any_clip = True
                ax.barh(y, 1.0, height=0.62, color=NEUTRAL, lw=0, zorder=2)
                ax.barh(y, end - 1.0, left=1.0, height=0.62, color=OPT_COLOR[opt],
                        lw=0, zorder=2)
                ax.plot([1.0, 1.0], [y - 0.31, y + 0.31], color="white", lw=0.8,
                        zorder=3)
                label = (f"{r['total']:.2f}×" if r["total"] < 1.995
                         else fmt_mult(r["total"]))
                if clipped:
                    ax.text(end, y, "»", ha="right", va="center", fontsize=10,
                            color="white", fontweight="bold", zorder=4)
                ax.annotate(label, (end, y), xytext=(4, 0), textcoords="offset points",
                            va="center", ha="left", fontsize=8, color=INK)
            ax.xaxis.set_major_locator(MaxNLocator(4, integer=xmax > 4))
            ax.grid(axis="x", visible=True)
            _study_title(ax, study)
        fig.supxlabel("Training FLOPs, in units of the model's forward+backward cost",
                      fontsize=9)
        handles = [Patch(facecolor=NEUTRAL, lw=0, label="model forward + backward (= 1)"),
                   Patch(facecolor=PALE, lw=0, label="optimizer (in its colour)")]
        if any_clip:
            handles.append(Line2D([], [], marker="$»$", color=INK2, ms=7, mew=0,
                                  linestyle="none",
                                  label="bar cut short (label = true total)"))
        _legend(fig, handles, ncol=len(handles))
        return _save(fig, outdir, "f7_compute_cost")


def fig_f8(df, logs, outdir) -> list[Path]:
    """F8 — rank of each optimizer across studies (bump chart)."""
    tab = rank_table(df)
    if tab.empty:
        return _skip("f8_rank_summary", "no completed fresh-seed runs")
    studies = [s for s in HEADLINE_STUDIES if s in set(tab["study"])]
    xpos = {s: i for i, s in enumerate(studies)}
    opts = [o for o in OPT_ORDER if o in set(tab["optimizer"])]
    with style():
        fig, axes = _subplots(1, 1, (5.0, 2.9))
        ax = axes[0][0]
        any_hollow = False
        for opt in opts:
            t = tab[tab["optimizer"] == opt].copy()
            t["x"] = t["study"].map(xpos)
            t = t.sort_values("x")
            c = OPT_COLOR[opt]
            ax.plot(t["x"], t["rank"], "-", color=c, lw=2.0, zorder=2)
            for _, r in t.iterrows():
                hollow = not r["separated"]
                any_hollow |= hollow
                ax.plot([r["x"]], [r["rank"]], marker=OPT_MARKER[opt],
                        mfc="white" if hollow else c, mec=c if hollow else "white",
                        mew=1.8 if hollow else 1.0, ms=8.5 if hollow else 9.5,
                        linestyle="none", zorder=3)
            last = t.iloc[-1]
            ax.annotate(OPT_LABEL[opt], (last["x"], last["rank"]), xytext=(10, 0),
                        textcoords="offset points", va="center", ha="left",
                        fontsize=9, color=INK)
        n = int(tab["rank"].max())
        ax.set_ylim(n + 0.45, 0.55)
        ax.set_yticks(range(1, n + 1))
        ax.set_yticklabels([f"{k}" + (" (best)" if k == 1 else "") for k in range(1, n + 1)])
        ax.set_xlim(-0.3, len(studies) - 1 + 1.25)
        ax.set_xticks(range(len(studies)))
        ax.set_xticklabels([STUDY_LABEL[s].replace(" ", "\n") for s in studies])
        for t in ax.get_xticklabels():
            t.set_color(INK)
        ax.set_ylabel("Rank by mean final validation loss")
        ax.grid(axis="x", visible=False)
        ax.spines["bottom"].set_bounds(0, len(studies) - 1)
        n_txt = seeds_text("f8_rank_summary", {f"{s}/{o}": n for s in studies
                                               for o, n in seed_counts(df, s).items()})
        handles = [Line2D([], [], marker="o", color=PALE, ms=9.5, linestyle="none",
                          label=f"gap to the next-better optimizer passes the "
                                f"{n_txt}-seed rule")]
        if any_hollow:
            handles.append(Line2D([], [], marker="o", mfc="white", mec=PALE, mew=1.8,
                                  ms=8.5, linestyle="none",
                                  label="not separated from the next-better optimizer "
                                        f"({n_txt}-seed heuristic, not a test)"))
        _legend(fig, handles, ncol=1)
        return _save(fig, outdir, "f8_rank_summary")


# ===========================================================================
# Appendix figures

#: y clip of A1 (loss minus sweep minimum, nats): keeps the near-optimum
#: region readable; points above it are drawn as ^ at the top edge.
A1_CLIP = 0.3


def fig_a1(df, logs, outdir) -> list[Path]:
    """A1 — loss vs learning rate for all optimizers (relative and raw LR)."""
    studies = [s for s in headline_studies(df)
               if any(not lr_sweep(df, s, o).empty for o in OPT_ORDER)]
    if not studies:
        return _skip("a1_lr_sensitivity", "no learning-rate sweeps")
    with style():
        fig, axes = _subplots(2, len(studies), (FULL_W, 4.5))
        used = []
        flags = dict(div=False, early=False, over=False)
        for j, study in enumerate(studies):
            a0, a1 = axes[0][j], axes[1][j]
            rel_all, raw_all, near, curves = [], [], [], []
            for opt in OPT_ORDER:
                sw = lr_sweep(df, study, opt)
                best_lr = an._best_lr(df, study, opt)
                ok = sw[sw["status"] == "completed"]
                if ok.empty or best_lr is None:
                    continue
                if opt not in used:
                    used.append(opt)
                d = (ok["loss"] - float(ok["loss"].min())).to_numpy()
                lrs = ok["lr"].to_numpy(dtype=float)
                rel = lrs / best_lr
                early = earlier_pass_divergences(df, study, opt)
                div = list(sw[sw["status"] == "diverged"]["lr"])
                curves.append((opt, lrs, d, best_lr, div, early))
                rel_all += list(sw["lr"] / best_lr) + [v / best_lr for v in early]
                raw_all += list(sw["lr"]) + early
                near += list(d[(rel >= 1 / 4.01) & (rel <= 4.01)])
            ytop = min(max(near + [1e-6]) * 1.2, A1_CLIP)
            for opt, lrs, d, best_lr, div, early in curves:
                c, mk = OPT_COLOR[opt], OPT_MARKER[opt]
                inr = d <= ytop
                for ax, xs in ((a0, lrs / best_lr), (a1, lrs)):
                    tr = ax.get_xaxis_transform()
                    ax.plot(xs, d, "-", color=c, zorder=3)
                    ax.plot(xs[inr], d[inr], linestyle="none", color=c, marker=mk, ms=5,
                            mew=0.7, zorder=4)
                    if (~inr).any():
                        flags["over"] = True
                        ax.plot(xs[~inr], [0.955] * int((~inr).sum()), "^", color=c,
                                ms=4.2, mew=0.6, transform=tr, zorder=5)
                    scale = (1.0 / best_lr) if ax is a0 else 1.0
                    for lr in div:
                        flags["div"] = True
                        ax.plot([lr * scale], [0.955], "x", color=c, mec=c, ms=6.5,
                                mew=1.8, transform=tr, zorder=6)
                    for lr in early:
                        flags["early"] = True
                        ax.plot([lr * scale], [0.955], "x", color=c, mec=c, ms=6.5,
                                mew=1.8, transform=tr, zorder=6)
            for ax in (a0, a1):
                ax.set_ylim(-0.04 * ytop, ytop)
                ax.yaxis.set_major_locator(MaxNLocator(4))
            a0.axvline(1.0, color=INK2, lw=0.8, zorder=1)
            a0.set_xscale("log", base=2)
            lo, hi = min(rel_all), max(rel_all)
            a0.set_xlim(lo / 1.5, hi * 1.5)
            ks = range(math.ceil(math.log2(lo) - 1e-9), math.floor(math.log2(hi) + 1e-9) + 1)
            step = 2 if len(ks) <= 9 else 3
            ticks = [2.0 ** k for k in ks if k % step == 0]
            a0.xaxis.set_major_locator(FixedLocator(ticks))
            a0.xaxis.set_major_formatter(FuncFormatter(
                lambda v, _: f"{v:g}" if v >= 1 else f"1/{1 / v:g}"))
            a0.xaxis.set_minor_locator(FixedLocator([2.0 ** k for k in ks]))
            a0.xaxis.set_minor_formatter(NullFormatter())
            _log_lr_axis(a1, raw_all)
            _study_title(a0, study)
            a0.set_xlabel("LR ÷ selected LR")
            a1.set_xlabel("Learning rate")
            if j == 0:
                a0.set_ylabel("Loss − sweep minimum\n(lower is better)")
                a1.set_ylabel("Loss − sweep minimum\n(lower is better)")
        handles = [_opt_handle(o) for o in used]
        if flags["over"]:
            handles.append(Line2D([], [], marker="^", color=PALE, ms=4.2, mew=0.6,
                                  linestyle="none", label="above the axis limit"))
        if flags["div"] and flags["early"]:
            label = "diverged (this sweep or an earlier tuning pass)"
        elif flags["early"]:
            label = "diverged in an earlier tuning pass"
        else:
            label = "diverged"
        if flags["div"] or flags["early"]:
            handles.append(Line2D([], [], marker="x", color=INK2, mec=INK2, ms=6.5,
                                  mew=1.8, linestyle="none", label=label))
        _legend(fig, handles, ncol=4 if len(handles) > 5 else len(handles))
        return _save(fig, outdir, "a1_lr_sensitivity")


def fig_a2(df, logs, outdir) -> list[Path]:
    """A2 — re-tuned vs inherited tuning track on MLP-256 / MLP-512 (seed 0)."""
    tab = track_comparison(df)
    if tab.empty:
        return _skip("a2_inherited_vs_retuned",
                     "no study with both a re-tuned and an inherited track")
    studies = [s for s in MLP_STUDIES if s in set(tab["study"])]
    opts = [o for o in OPT_ORDER if o in set(tab["optimizer"])]
    with style():
        fig, axes = _subplots(1, len(studies), (FULL_W, 2.6))
        for j, (ax, study) in enumerate(zip(axes[0], studies)):
            t = tab[tab["study"] == study].set_index("optimizer")
            pos = _opt_rows(ax, opts, labels=(j == 0))
            vals = list(t["retuned_loss"]) + list(t["inherited_loss"])
            lo, hi = min(vals), max(vals)
            span = max(hi - lo, 1e-9)
            ax.set_xlim(lo - 0.09 * span, hi + 0.09 * span)
            for opt in opts:
                if opt not in t.index:
                    continue
                r, y, c, mk = t.loc[opt], pos[opt], OPT_COLOR[opt], OPT_MARKER[opt]
                ax.plot([r["inherited_loss"], r["retuned_loss"]], [y, y], color=c,
                        lw=1.6, zorder=2)
                ax.plot([r["inherited_loss"]], [y], marker=mk, mfc="white", mec=c,
                        mew=1.4, ms=10.5, linestyle="none", zorder=3)
                ax.plot([r["retuned_loss"]], [y], marker=mk, color=c, ms=6.5,
                        linestyle="none", zorder=4)
                if math.isclose(r["inherited_lr"], r["retuned_lr"]):
                    txt = f"LR {fmt_num(r['retuned_lr'])} (both)"
                else:
                    txt = f"LR {fmt_num(r['inherited_lr'])} → {fmt_num(r['retuned_lr'])}"
                ax.text(1.03, y, txt, transform=ax.get_yaxis_transform(), ha="left",
                        va="center", fontsize=8, color=INK)
            thr = noise_threshold(df, study)
            if np.isfinite(thr):
                x0 = lo
                ax.plot([x0, x0 + thr], [len(opts) - 0.52] * 2, color=INK2, lw=2.2,
                        solid_capstyle="butt", zorder=3)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            _study_title(ax, study)
        fig.supxlabel("Final validation loss of the tuning winner, seed 0 "
                      "(lower is better)", fontsize=9)
        handles = [
            Line2D([], [], marker="o", mfc="white", mec=PALE, mew=1.4, ms=10.5,
                   linestyle="none",
                   label="inherited MLP-128 settings + LR × WD grid"),
            Line2D([], [], marker="o", color=PALE, ms=6.5, linestyle="none",
                   label="re-tuned (coordinate descent)"),
            Line2D([], [], color=INK2, lw=2.2,
                   label="scale: 2 × pilot seed std"),
        ]
        _legend(fig, handles, ncol=3)
        return _save(fig, outdir, "a2_inherited_vs_retuned")


def fig_a3(df, logs, outdir) -> list[Path]:
    """A3 — robustness: faithful settings, init ablation, half-LR 2x AdamW."""
    fa, ia, hl = faithful_pairs(df), init_ablation_runs(df), halflr_pairs(df)
    blocks = []
    if not fa.empty:
        blocks.append("a")
    else:
        _note("a3 (a): no faithful-settings reruns — panel row skipped")
    if not ia.empty:
        blocks.append("b")
    else:
        _note("a3 (b): no init-ablation runs — panel row skipped")
    if not hl.empty:
        blocks.append("c")
    else:
        _note("a3 (c): no half-LR 2x AdamW runs — panel row skipped")
    if not blocks:
        return _skip("a3_robustness", "no faithful / init-ablation / half-LR runs")
    present = set(fa["study"]) | set(ia["study"]) | set(hl["study"])
    studies = [s for s in HEADLINE_STUDIES if s in present]
    heights = {"a": 1.25, "b": 0.95, "c": 0.95}
    with style():
        fig, axes = _subplots(len(blocks), len(studies),
                              (FULL_W, sum(heights[b] for b in blocks) + 1.45),
                              height_ratios=[heights[b] for b in blocks])

        def xpad(ax, vals, left=0.12, right=0.12):
            lo, hi = min(vals), max(vals)
            span = max(hi - lo, 1e-9)
            ax.set_xlim(lo - left * span, hi + right * span)
            ax.xaxis.set_major_locator(MaxNLocator(3, min_n_ticks=3))

        def rows(ax, labels, show):
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels if show else [])
            ax.set_ylim(len(labels) - 0.4, -0.6)
            ax.grid(axis="y", visible=False)
            ax.tick_params(axis="y", length=0)
            for t in ax.get_yticklabels():
                t.set_color(INK)

        for i, b in enumerate(blocks):
            for j, study in enumerate(studies):
                ax = axes[i][j]
                if i == 0:
                    _study_title(ax, study)
                if b == "a":
                    t = fa[fa["study"] == study].set_index("optimizer")
                    opts = [o for o in OPT_ORDER if o in set(fa["optimizer"])]
                    rows(ax, [OPT_LABEL[o] for o in opts], j == 0)
                    vals = []
                    for y, opt in enumerate(opts):
                        if opt not in t.index:
                            continue
                        r, c, mk = t.loc[opt], OPT_COLOR[opt], OPT_MARKER[opt]
                        ax.plot([r["lo"], r["hi"]], [y, y], color=c, lw=7, alpha=0.22,
                                solid_capstyle="butt", zorder=1)
                        ax.plot([r["standard"], r["faithful"]], [y, y], color=c, lw=1.2,
                                zorder=2)
                        ax.plot([r["faithful"]], [y], marker=mk, mfc="white", mec=c,
                                mew=1.4, ms=9.5, linestyle="none", zorder=3)
                        ax.plot([r["standard"]], [y], marker=mk, color=c, ms=6,
                                linestyle="none", zorder=4)
                        vals += [r["lo"], r["hi"], r["faithful"], r["standard"]]
                    if vals:
                        xpad(ax, vals)
                    if j == 0:
                        ax.set_ylabel("(a) Faithful\nsettings", fontweight="bold")
                elif b == "b":
                    t = ia[ia["study"] == study]
                    variants = [("standard", "Muon"), ("stiefel_init", "Muon, Stiefel init")]
                    rows(ax, [v[1] for v in variants], j == 0)
                    opt = t["optimizer"].iloc[0] if not t.empty else "muon"
                    c, mk = OPT_COLOR[opt], OPT_MARKER[opt]
                    for y, (v, _) in enumerate(variants):
                        g = t[t["variant"] == v]
                        if g.empty:
                            continue
                        hollow = v != "standard"
                        ax.plot([g["loss"].mean()], [y], marker=mk,
                                mfc="white" if hollow else c, mec=c if hollow else "white",
                                mew=1.4 if hollow else 1.0, ms=9.5 if hollow else 8.5,
                                linestyle="none", zorder=3)
                        ax.plot(g["loss"], [y] * len(g), "o", color=c, ms=3.6,
                                mec="white", mew=0.6, zorder=4)
                    if not t.empty:
                        xpad(ax, list(t["loss"]))
                    if j == 0:
                        ax.set_ylabel("(b) Init\nablation", fontweight="bold")
                else:
                    t = hl[hl["study"] == study]
                    rows(ax, ["AdamW 2×", "AdamW 2×, half LR"], j == 0)
                    c, mk = OPT_COLOR["adamw"], OPT_MARKER["adamw"]
                    a2 = df[(df["study"] == study) & (df["optimizer"] == "adamw")
                            & (df["phase"] == "budget_constant")
                            & (df["status"] == "completed")]["final_val_loss"]
                    vals = list(a2)
                    if len(a2):
                        ax.plot(a2, [0] * len(a2), "o", color=c, ms=3.6, mec="white",
                                mew=0.6, zorder=4, alpha=0.55)
                    for _, r in t.iterrows():
                        ax.plot([r["full_lr"], r["half_lr"]], [0, 1], color=c, lw=1.2,
                                zorder=2)
                        ax.plot([r["full_lr"]], [0], marker=mk, color=c, ms=6,
                                linestyle="none", zorder=5)
                        ax.plot([r["half_lr"]], [1], marker=mk, mfc="white", mec=c,
                                mew=1.4, ms=7, linestyle="none", zorder=5)
                        vals += [r["full_lr"], r["half_lr"]]
                    if vals:
                        xpad(ax, vals)
                    if j == 0:
                        ax.set_ylabel("(c) AdamW\n2× budget", fontweight="bold")
        fig.supxlabel("Final validation loss (lower is better)", fontsize=9)
        pairs = ({(r["study"], r["optimizer"]) for _, r in fa.iterrows()}
                 | {(r["study"], r["optimizer"]) for _, r in ia.iterrows()})
        n_txt = seeds_text("a3_robustness", {
            f"{s}/{o}": int(len(an._fresh(df, s, o))) for s, o in sorted(pairs)})
        handles = [
            Line2D([], [], marker="o", color=PALE, ms=6, linestyle="none",
                   label="standard run (a, c: the paired seed; b: mean)"),
            Line2D([], [], marker="o", mfc="white", mec=PALE, mew=1.4, ms=9,
                   linestyle="none", label="variant (faithful / Stiefel init / half LR)"),
            Line2D([], [], marker="o", color=PALE, ms=3.6, mec="white", mew=0.6,
                   linestyle="none", label="one seed"),
            Patch(facecolor=PALE, alpha=0.4, lw=0,
                  label=f"(a) min–max of the {n_txt} standard seeds"),
        ]
        _legend(fig, handles, ncol=2)
        return _save(fig, outdir, "a3_robustness")


def fig_a4(df, logs, outdir) -> list[Path]:
    """A4 — final validation loss vs test accuracy for every 1x MLP run."""
    m = (df["study"].isin(MLP_STUDIES) & (df["status"] == "completed")
         & (df["phase"] != "smoke") & df["final_val_loss"].notna()
         & df["final_test_acc"].notna() & df["optimizer"].isin(OPT_ORDER))
    if "budget_multiplier" in df.columns:
        m &= df["budget_multiplier"] == 1.0
    runs = df[m]
    studies = [s for s in MLP_STUDIES if s in set(runs["study"])]
    if not studies:
        return _skip("a4_loss_vs_accuracy", "no completed 1x MLP runs with test accuracy")
    with style():
        fig, axes = _subplots(1, len(studies), (FULL_W, 2.9), sharey=True)
        for j, (ax, study) in enumerate(zip(axes[0], studies)):
            r = runs[runs["study"] == study]
            for opt in OPT_ORDER:
                g = r[r["optimizer"] == opt]
                tun, fresh = g[g["phase"] != "fresh_seed"], g[g["phase"] == "fresh_seed"]
                ax.plot(tun["final_val_loss"], 100 * tun["final_test_acc"],
                        linestyle="none", marker=OPT_MARKER[opt], color=OPT_COLOR[opt],
                        ms=3.6, mew=0, alpha=0.38, zorder=2)
                ax.plot(fresh["final_val_loss"], 100 * fresh["final_test_acc"],
                        linestyle="none", marker=OPT_MARKER[opt], color=OPT_COLOR[opt],
                        ms=4.8, mec=INK, mew=0.5, zorder=3)
            _study_title(ax, study)
            ax.xaxis.set_major_locator(MaxNLocator(5))
        acc = runs["final_test_acc"].to_numpy(dtype=float)
        p = float(np.median(acc))
        se = 100 * math.sqrt(p * (1 - p) / N_TEST_IMAGES)
        y0, y1 = axes[0][0].get_ylim()
        yb = y1 - 0.10 * (y1 - y0)
        for ax in axes[0]:
            ax.errorbar([0.93], [yb], yerr=[se], transform=ax.get_yaxis_transform(),
                        color=INK, lw=1.2, capsize=3, capthick=1.2, zorder=5)
        axes[0][0].set_ylim(y0, y1)
        axes[0][0].set_ylabel("Test accuracy (%)")
        fig.supxlabel("Final validation loss (lower is better)", fontsize=9)
        handles = [_opt_handle(o, line=False) for o in OPT_ORDER
                   if o in set(runs["optimizer"])]
        handles += [
            Line2D([], [], marker="o", color=PALE, ms=4.8, mec=INK, mew=0.5,
                   linestyle="none", label="fresh-seed run (dark edge)"),
            Line2D([], [], marker="o", color=PALE, ms=3.6, mew=0, alpha=0.6,
                   linestyle="none", label="tuning / diagnostic run"),
            Line2D([], [], marker="|", color=INK, mec=INK, ms=9, mew=1.2,
                   linestyle="none",
                   label=f"±1 SE of accuracy ({se:.2f} pt, {N_TEST_IMAGES:,} test "
                         f"images)"),
        ]
        _legend(fig, handles, ncol=4 if len(handles) <= 8 else 3)
        return _save(fig, outdir, "a4_loss_vs_accuracy")


def fig_a5(df, logs, outdir) -> list[Path]:
    """A5 — Manifold Muon audit: MLP-128 ablations and final spectral norms."""
    root = Path(df.attrs.get("root", "."))
    aud = manifold_audit_rows(root)
    sp = spectral_by_layer(logs, df, "dense")
    if aud.empty:
        _note("a5 (a): logs/diagnostics/manifold_audit/summary.jsonl not found — "
              "panel skipped")
    if sp.empty:
        _note("a5 (b): no dense spectral norms in the fresh-seed logs — panel skipped")
    if aud.empty and sp.empty:
        return _skip("a5_manifold_audit", "no audit summary and no spectral norms")
    panels = [p for p, d in (("a", aud), ("b", sp)) if not d.empty]
    with style():
        fig, axes = _subplots(1, len(panels), (FULL_W, 3.9),
                              width_ratios=[1.75, 1.0][:len(panels)]
                              if len(panels) == 2 else None)
        handles = []
        for ax, p in zip(axes[0], panels):
            if p == "a":
                study = "mlp128"
                rows = aud.to_dict("records")
                fa = faithful_pairs(df)
                fa = fa[(fa["study"] == study) & (fa["optimizer"] == "manifold")]
                for _, r in fa.iterrows():
                    rows.append({"name": "faithful", "optimizer": "manifold",
                                 "label": f"Cap 100 (faithful rerun, seed {r['seed']})",
                                 "loss": r["faithful"], "status": "completed"})
                refs = [r for r in rows if r["name"] == "muon_seed0_reference"]
                rows = [r for r in rows if r["name"] != "muon_seed0_reference"]
                rows.sort(key=lambda r: (r["status"] != "completed",
                                         r["loss"] if r["status"] == "completed" else 0))
                ok = [r["loss"] for r in rows if r["status"] == "completed"]
                # reference lines (seed 0, like the ablations)
                ref_vals = {}
                if refs and refs[0]["status"] == "completed":
                    ref_vals["muon"] = refs[0]["loss"]
                acc = accepted_config(df, study, "adamw")
                cd = df[(df["study"] == study) & (df["optimizer"] == "adamw")
                        & (df["phase"] == "cd") & (df["status"] == "completed")]
                hit = [r for r in cd.to_dict("records") if _hypers(r, "adamw") == acc]
                if hit:
                    ref_vals["adamw"] = float(hit[0]["final_val_loss"])
                allv = ok + list(ref_vals.values())
                lo, hi = min(allv), max(allv)
                span = max(hi - lo, 1e-9)
                x1 = hi + 0.10 * span
                ax.set_xlim(lo - 0.06 * span, x1)
                for opt, v in ref_vals.items():
                    ax.axvline(v, color=OPT_COLOR[opt], lw=1.2, ls=(0, (4, 2)), zorder=1)
                    handles.append(Line2D([], [], color=OPT_COLOR[opt], lw=1.2,
                                          ls=(0, (4, 2)),
                                          label=f"{OPT_LABEL[opt]}, frozen-protocol "
                                                f"config (seed 0)"))
                for y, r in enumerate(rows):
                    c, mk = OPT_COLOR[r["optimizer"]], OPT_MARKER[r["optimizer"]]
                    if r["status"] == "completed":
                        ax.plot([lo - 0.06 * span, r["loss"]], [y, y], color=GRID,
                                lw=0.8, zorder=1)
                        ax.plot([r["loss"]], [y], marker=mk, color=c, ms=6.5,
                                linestyle="none", zorder=3)
                    else:
                        ax.plot([0.965], [y], "x", color=INK, mec=INK, ms=6, mew=1.5,
                                transform=ax.get_yaxis_transform(), zorder=3)
                        ax.text(0.93, y, "diverged", transform=ax.get_yaxis_transform(),
                                ha="right", va="center", fontsize=8, color=INK2)
                ax.set_yticks(range(len(rows)))
                ax.set_yticklabels([r["label"] for r in rows])
                for t in ax.get_yticklabels():
                    t.set_color(INK)
                ax.set_ylim(len(rows) - 0.4, -0.6)
                ax.grid(axis="y", visible=False)
                ax.tick_params(axis="y", length=0)
                ax.set_xlabel("Final validation loss (lower is better)")
                other = sorted({int(r["seed"]) for _, r in fa.iterrows()} - {0})
                ax.set_title(f"(a) {STUDY_LABEL[study]} ablations "
                             + ("(seed 0 unless noted)" if other else "(seed 0)"))
                used = [o for o in OPT_ORDER if o in {r["optimizer"] for r in rows}]
                handles = [Line2D([], [], marker=OPT_MARKER[o], color=OPT_COLOR[o],
                                  ms=6.5, linestyle="none",
                                  label=f"{OPT_LABEL[o]} run") for o in used] + handles
            else:
                opts = [o for o in OPT_ORDER if o in set(sp["optimizer"])]
                n_sp = seeds_text("a5_manifold_audit (b)",
                                  spectral_seed_counts(logs, df, "dense"))
                rng = np.random.default_rng(0)
                for x, opt in enumerate(opts):
                    v = sp[sp["optimizer"] == opt]["sigma"].to_numpy()
                    jit = rng.uniform(-0.24, 0.24, size=v.size)
                    ax.plot(x + jit, v, linestyle="none", marker=OPT_MARKER[opt],
                            color=OPT_COLOR[opt], ms=4.2, mew=0.5, alpha=0.8, zorder=3)
                    ax.plot([x - 0.34, x + 0.34], [np.median(v)] * 2, color=INK, lw=1.4,
                            zorder=4)
                ax.axhline(1.0, color=INK2, lw=0.8, zorder=1)
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                ax.yaxis.set_minor_formatter(NullFormatter())
                ax.set_xticks(range(len(opts)))
                ax.set_xticklabels([OPT_LABEL[o] for o in opts], rotation=30,
                                   ha="right", rotation_mode="anchor")
                for t in ax.get_xticklabels():
                    t.set_color(INK)
                ax.set_xlim(-0.6, len(opts) - 0.4)
                ax.grid(axis="x", visible=False)
                ax.set_ylabel("Final σ$_{max}$ per matrix (log scale)")
                ax.set_title("(b) Dense transformer")
                handles.append(Line2D([], [], color=INK, lw=1.4,
                                      label=f"(b) median over matrices; one point per "
                                            f"matrix, mean of {n_sp} seeds"))
        _legend(fig, handles, ncol=2)
        return _save(fig, outdir, "a5_manifold_audit")


def fig_a6(df, logs, outdir) -> list[Path]:
    """A6 — max attention logit over training (dense)."""
    d = attn_logit_bands(logs, df, "dense")
    if d.empty:
        return _skip("a6_attention_logits", "no dense attention-logit logs")
    opts = [o for o in OPT_ORDER if o in set(d["optimizer"])]
    with style():
        fig, axes = _subplots(1, 1, (COL_W, 3.0))
        ax = axes[0][0]
        for opt in opts:
            g = d[d["optimizer"] == opt].sort_values("x")
            ax.fill_between(g["x"] / 1e6, g["lo"], g["hi"], color=OPT_COLOR[opt],
                            alpha=0.18, lw=0, zorder=2)
            ax.plot(g["x"] / 1e6, g["mean"], color=OPT_COLOR[opt], lw=1.8, zorder=3)
            ax.plot([g["x"].iloc[-1] / 1e6], [g["mean"].iloc[-1]],
                    marker=OPT_MARKER[opt], color=OPT_COLOR[opt], linestyle="none",
                    ms=5.5, mew=0.8, zorder=4, clip_on=False)
        ax.set_xlabel("Tokens seen (millions)")
        n_txt = seeds_text("a6_attention_logits", {
            o: int(d[d["optimizer"] == o]["n_seeds"].min()) for o in opts})
        ax.set_ylabel(f"Max attention logit\n(mean, min–max over {n_txt} seeds)")
        ax.set_xlim(0, float(d["x"].max()) / 1e6 * 1.03)
        _legend(fig, [_opt_handle(o) for o in opts], ncol=3)
        return _save(fig, outdir, "a6_attention_logits")


def fig_a7(df, logs, outdir) -> list[Path]:
    """A7 — final sigma_max by matrix type and layer (dense)."""
    sp = spectral_by_layer(logs, df, "dense")
    sp = sp[sp["layer"].notna()] if not sp.empty else sp
    kinds = [k for k in an._MODULE_ORDER if k in set(sp["kind"])] if not sp.empty else []
    if not kinds:
        return _skip("a7_spectral_norms_by_layer", "no dense spectral norms by layer")
    opts = [o for o in OPT_ORDER if o in set(sp["optimizer"])]
    if "adamw" not in opts:
        _note("a7: AdamW logs no spectral norms (empty dict in its final record) — "
              "not drawn")
    titles = {"q": "Query", "k": "Key", "v": "Value", "o": "Attention out",
              "gate": "MLP gate", "up": "MLP up", "down": "MLP down"}
    ncols = 4
    nrows = math.ceil(len(kinds) / ncols)
    with style():
        fig, axes = _subplots(nrows, ncols, (FULL_W, 1.95 * nrows + 0.55), sharey=True,
                              sharex=True)
        flat = list(axes.ravel())
        layers = sorted(int(v) for v in sp["layer"].unique())
        for ax, kind in zip(flat, kinds):
            for opt in opts:
                g = sp[(sp["kind"] == kind) & (sp["optimizer"] == opt)].sort_values("layer")
                ax.plot(g["layer"], g["sigma"], "-", color=OPT_COLOR[opt],
                        marker=OPT_MARKER[opt], ms=5, mew=0.7, zorder=3)
            ax.axhline(1.0, color=INK2, lw=0.8, zorder=1)
            ax.set_yscale("log")
            ax.set_title(titles.get(kind, kind))
            ax.set_xticks(layers)
            ax.set_xlim(min(layers) - 0.4, max(layers) + 0.4)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
        for ax in flat[len(kinds):]:
            ax.axis("off")
        for k, ax in enumerate(flat[:len(kinds)]):
            below = k + ncols
            if below >= len(kinds):
                ax.tick_params(labelbottom=True)
        fig.supxlabel("Layer", fontsize=9)
        n_txt = seeds_text("a7_spectral_norms_by_layer",
                           spectral_seed_counts(logs, df, "dense"))
        fig.supylabel(f"Final σ$_{{max}}$, mean over {n_txt} seeds (log scale)",
                      fontsize=9)
        handles = [_opt_handle(o) for o in opts]
        if len(flat) > len(kinds):
            leg = flat[-1].legend(handles=handles, loc="center", ncol=1,
                                  title="Dense transformer", title_fontsize=9)
            for t in leg.get_texts():
                t.set_color(INK)
        else:
            _legend(fig, handles, ncol=len(handles))
        return _save(fig, outdir, "a7_spectral_norms_by_layer")


def fig_a8(df, logs, outdir) -> list[Path]:
    """A8 — learning-rate range trained without divergence, per study."""
    studies = [s for s in headline_studies(df)
               if any(not lr_stability(df, s, o).empty for o in OPT_ORDER)]
    if not studies:
        return _skip("a8_divergence", "no tuning runs")
    opts = [o for o in OPT_ORDER
            if any(not lr_stability(df, s, o).empty for s in studies)]
    ncols = 2 if len(studies) > 1 else 1
    nrows = math.ceil(len(studies) / ncols)
    # every panel prints its own row labels, so a study without runs of an
    # optimizer simply has no row for it (rows are shared only while every
    # study has the same optimizers)
    all_opts = opts
    panel_opts = [[o for o in all_opts if not lr_stability(df, s, o).empty]
                  for s in studies]
    with style():
        fig, axes = _subplots(nrows, ncols, (FULL_W, 1.85 * nrows + 0.75), sharex=True,
                              sharey=all(o == panel_opts[0] for o in panel_opts))
        flat = list(axes.ravel())
        all_lrs = []
        flags = dict(high=False, div=False)
        for k, (ax, study) in enumerate(zip(flat, studies)):
            opts = panel_opts[k]
            pos = _opt_rows(ax, opts, labels=True)
            for opt in opts:
                t = lr_stability(df, study, opt)
                if t.empty:
                    continue
                y, c, mk = pos[opt], OPT_COLOR[opt], OPT_MARKER[opt]
                all_lrs += list(t["lr"])
                ok = t[t["state"] == "ok"]
                if not ok.empty:
                    ax.plot([ok["lr"].min(), ok["lr"].max()], [y, y], color=c, lw=6,
                            alpha=0.3, solid_capstyle="round", zorder=2)
                    ax.plot(ok["lr"], [y] * len(ok), "o", color=c, ms=3.4, mew=0,
                            zorder=3)
                hi = t[t["state"] == "high_loss"]
                if not hi.empty:
                    flags["high"] = True
                    ax.plot(hi["lr"], [y] * len(hi), "o", mfc="white", mec=c, mew=1.0,
                            ms=3.8, zorder=3)
                dv = t[t["state"] == "diverged"]
                if not dv.empty:
                    flags["div"] = True
                    ax.plot(dv["lr"], [y] * len(dv), "x", color=INK, mec=INK, ms=6, mew=1.5,
                            zorder=4)
                b = t[t["best"]]
                if not b.empty:
                    ax.plot(b["lr"], [y] * len(b), marker=mk, color=c, ms=8, mec=INK,
                            mew=0.9, linestyle="none", zorder=5)
            _study_title(ax, study)
        for ax in flat[len(studies):]:
            ax.set_visible(False)
        _log_lr_axis(flat[0], all_lrs)
        flat[0].xaxis.set_minor_locator(FixedLocator([]))
        fig.supxlabel("Learning rate (log scale)", fontsize=9)
        handles = [
            Patch(facecolor=PALE, alpha=0.5, lw=0,
                  label="range trained without divergence (dots: tested LRs)"),
            Line2D([], [], marker="o", color=PALE, mec=INK, mew=0.9, ms=8,
                   linestyle="none", label="selected LR"),
        ]
        if flags["high"]:
            handles.append(Line2D([], [], marker="o", mfc="white", mec=PALE, mew=1.0,
                                  ms=3.8, linestyle="none",
                                  label="completed, but > best + 0.5 nats"))
        if flags["div"]:
            handles.append(Line2D([], [], marker="x", color=INK, mec=INK, ms=6, mew=1.5,
                                  linestyle="none", label="diverged (run stopped)"))
        _legend(fig, handles, ncol=2)
        return _save(fig, outdir, "a8_divergence")


# ===========================================================================
# MoE figures (skipped until MoE fresh seeds exist)

def moe_ready(df: pd.DataFrame) -> bool:
    """True once the MoE study has completed fresh-seed runs."""
    return bool(df is not None and not df.empty and not an._fresh(df, "moe").empty)


def fig_moe(df, logs, outdir) -> list[Path]:
    """MoE figures (expert entropy, dense-vs-MoE interaction): not drawn until
    MoE fresh seeds exist — and not specified beyond that yet."""
    if not moe_ready(df):
        return _skip("moe_figures", "no MoE fresh-seed runs yet (only smoke runs); "
                                    "expert-entropy and dense-vs-MoE figures not drawn")
    return _skip("moe_figures", "MoE fresh seeds exist, but the MoE figures are not "
                                "specified yet (docs/figures_spec.md)")


# ===========================================================================
# Entry point

def figure_plan(df: pd.DataFrame) -> list[tuple[str, object]]:
    """(figure id, callable(df, logs, outdir)) for every figure, in order."""
    plan: list[tuple[str, object]] = [
        ("f1_final_loss", fig_f1), ("f2_speedup", fig_f2),
        ("f3_budget_curve", fig_f3), ("f4_loss_curves_tokens", fig_f4),
        ("f4b_loss_curves_flops", fig_f4b), ("f5_width_transfer", fig_f5),
        ("f5b_optimal_lr_vs_width", fig_f5b),
    ]
    for study in HEADLINE_STUDIES:
        if df is not None and not df.empty and an._has_cd_rows(df, study):
            plan.append((f"f6_hparam_sensitivity_{study}",
                         lambda d, l, o, s=study: fig_f6(d, l, o, s)))
    plan += [
        ("f7_compute_cost", fig_f7), ("f8_rank_summary", fig_f8),
        ("a1_lr_sensitivity", fig_a1), ("a2_inherited_vs_retuned", fig_a2),
        ("a3_robustness", fig_a3), ("a4_loss_vs_accuracy", fig_a4),
        ("a5_manifold_audit", fig_a5), ("a6_attention_logits", fig_a6),
        ("a7_spectral_norms_by_layer", fig_a7), ("a8_divergence", fig_a8),
        ("moe_figures", fig_moe),
    ]
    return plan


def build_all(df: pd.DataFrame, logs: pd.DataFrame | None, outdir: Path | str,
              out=None) -> dict[str, list[Path]]:
    """Build every figure; print one line per figure (written or skipped)."""
    out = out or sys.stdout
    results: dict[str, list[Path]] = {}
    for fig_id, fn in figure_plan(df):
        _NOTES.clear()
        SKIP_REASONS.pop(fig_id, None)
        if df is None or df.empty:
            paths = _skip(fig_id, "no result files")
        else:
            paths = fn(df, logs, outdir)
        results[fig_id] = paths
        notes = "".join(f" [note: {n}]" for n in _NOTES)
        if paths:
            print(f"{fig_id}: written ({', '.join(p.name for p in paths)}){notes}",
                  file=out)
        else:
            print(f"{fig_id}: skipped — {SKIP_REASONS.get(fig_id, 'no data')}{notes}",
                  file=out)
    return results


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    root = Path(argv[0]) if argv else Path(".")
    outdir = Path(argv[1]) if len(argv) > 1 else root / "figures"
    df = an.load_results(root)
    logs = an.load_logs(root)
    build_all(df, logs, outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
